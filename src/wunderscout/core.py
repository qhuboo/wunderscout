from pathlib import Path
import logging
import supervision as sv
import numpy as np
import cv2

from wunderscout.types import ClassId, Frames

from .models import Models
from .geometry import PitchMapper
from .teams import TeamClassifier

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, models: Models):
        self._models = models
        self._mapper = PitchMapper()
        self._classifier = TeamClassifier()
        self._is_calibrated = False

    def _validate_video(
        self, video_path: str | Path
    ) -> tuple[int, float, tuple[int, int]]:
        """
        Validate video can be opened and read.

        Args:
            video_path: Path to source video file.

        Returns:
            tuple: (frame_count, fps, (width, height))

        Raises:
            FileNotFoundError: If file is not found from video_path.
            ValueError: If video file cannot be opened.
            OSError: If video file frame reading fails.
        """
        logger.info("Validating video...")

        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Invalid or unsupported video format: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Test reading first frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise OSError(
                f"Cannot read video data (corrupted or incomplete): {video_path}"
            )

        logger.info(
            f"Video validated: {frame_count} frames, {fps:.1f} fps, {width}x{height}"
        )

        return frame_count, fps, (width, height)

    def _calibrate(self, video_path: str | Path) -> None:
        """
        Calibrate team classifier using player detections.

        Args:
            video_path: Path to source video.

        Raises:
            RuntimeError: If no player crops found or calibration fails.
        """
        logger.info("Starting calibration...")

        crops = self._models._get_calibration_crops(
            video_path, class_id=ClassId.PLAYER.value, stride=10
        )

        if not crops:
            raise RuntimeError("No player detections found for calibration.")

        logger.info(f"Found {len(crops)} player crops for calibration.")

        embeddings = self._models._get_embeddings(crops)
        logger.debug(f"Generating embeddings shape: {embeddings.shape}")

        self._classifier.fit(embeddings)
        self._is_calibrated = True

        logger.info("Calibration successful.")

    def run(self, video_path, output_dir: str | Path | None = None) -> Frames:
        """
        Run detection and tracking on video.

        Args:
            video_path: Path to input video file.
            output_dir: Optional dir path to save annotated video.
        Returns:
            Frames: Detection results for all frames.

        Raises:
            VideoProcessingError: If video cannot be processed.
            CalibrationError: If team calibration fails.
        """
        video_path = Path(video_path)

        if output_dir is not None:
            output_dir = Path(output_dir)

            if output_dir.is_file():
                raise ValueError(
                    f"output_dir must be a directory, not a file: {output_dir}."
                )

        logger.info(f"Processing video: {video_path}")

        frame_count, _, _ = self._validate_video(video_path)

        # Calibrate classifier
        self._calibrate(video_path)

        # Start tracker
        tracker = sv.ByteTrack()
        frames = []

        # 3. Main Processing Loop
        logger.info(f"Starting processing: {video_path}")
        frame_generator = sv.get_video_frames_generator(str(video_path))
        for frame_idx, frame in enumerate(frame_generator):
            logger.info(f"Processing frame {frame_idx + 1}/{frame_count}")
            # --- A. DETECTION ---
            all_dets = self._models._detect_players(frame)
            f_res = self._models._detect_field(frame)

            logger.debug("Frame %d: Initial detections: %d", frame_idx, len(all_dets))

            # --- B. FIELD HOMOGRAPHY ---
            H = None
            if f_res.keypoints is not None and len(f_res.keypoints.xy) > 0:
                H = self._mapper.get_matrix(
                    f_res.keypoints.xy[0].cpu().numpy(),
                    f_res.keypoints.conf[0].cpu().numpy(),
                )
            else:
                H = self._mapper.last_h

            logger.debug(f"H: {H}")

            # --- C. SEPARATE BALL & OTHERS ---
            ball_detections = all_dets[all_dets.class_id == ClassId.BALL.value]
            other_detections = all_dets[all_dets.class_id != ClassId.BALL.value]
            other_detections = other_detections.with_nms(threshold=0.5)

            # --- D. TRACKING ---
            tracked_objects = tracker.update_with_detections(other_detections)

            # Split tracked objects
            tracked_players = tracked_objects[
                tracked_objects.class_id == ClassId.PLAYER.value
            ]
            tracked_gks = tracked_objects[
                tracked_objects.class_id == ClassId.GOALKEEPER.value
            ]

            # Pad ball_detections with tracker_ids to avoid error on merge
            ball_detections.tracker_id = np.array(
                [-1] * len(ball_detections), dtype=int
            )

            # --- E. TEAM CLASSIFICATION ---

            # 1. Players
            if len(tracked_players) > 0:
                p_crops = [sv.crop_image(frame, xyxy) for xyxy in tracked_players.xyxy]
                p_pil = [sv.cv2_to_pillow(c) for c in p_crops]
                p_embeddings = self._models._get_embeddings(p_pil)

                final_team_ids = self._classifier.get_consensus_teams(
                    tracked_players.tracker_id, p_embeddings
                )

                tracked_players.data["team_id"] = np.array(final_team_ids)

            else:
                tracked_players.data["team_id"] = np.array([], dtype=int)

            # 2. Goalkeepers
            if len(tracked_gks) > 0 and len(tracked_players) > 0:
                tracked_gks.data["team_id"] = (
                    self._classifier.resolve_goalkeepers_team_id(
                        tracked_players, tracked_gks
                    )
                )

            else:
                tracked_gks.data["team_id"] = np.array(
                    [-1] * len(tracked_gks), dtype=int
                )

            # 3. Ball
            ball_detections.data["team_id"] = np.array(
                [-1] * len(ball_detections), dtype=int
            )

            # --- F. DATA STORAGE ---
            data_targets = sv.Detections.merge([tracked_players, tracked_gks])

            if len(data_targets) > 0:
                if H is not None:
                    data_targets.data["pitch_coordinates"] = self._mapper.transform(
                        data_targets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER),
                        H,
                    )

                else:
                    data_targets.data["pitch_coordinates"] = np.full(
                        (len(data_targets), 2), np.nan
                    )
            else:
                data_targets.data["pitch_coordinates"] = np.full((0, 2), np.nan)

            if len(ball_detections) > 0:
                if H is not None:
                    ball_detections.data["pitch_coordinates"] = self._mapper.transform(
                        ball_detections.get_anchors_coordinates(sv.Position.CENTER), H
                    )
                else:
                    ball_detections.data["pitch_coordinates"] = np.full(
                        (len(ball_detections), 2), np.nan
                    )
            else:
                ball_detections.data["pitch_coordinates"] = np.full((0, 2), np.nan)

            merged = sv.Detections.merge([data_targets, ball_detections])
            frames.append(merged)

            # --- G. OPTIONAL VIDEO ANNOTATION ---
            # Something is really wrong with this new detection, its absolutely awful. Ill return to this later.
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)

                orig_path = Path(video_path)
                new_filename = f"{orig_path.stem}_annotated{orig_path.suffix}"
                final_video_file = output_dir / new_filename

                video_info = sv.VideoInfo.from_video_path(str(video_path))
                render_generator = sv.get_video_frames_generator(str(video_path))

                # Define team colors
                TEAM_COLORS = sv.ColorPalette(
                    [
                        sv.Color.from_hex("#FF6B6B"),  # Team 0 - Red
                        sv.Color.from_hex("#4ECDC4"),  # Team 1 - Teal
                        sv.Color.from_hex("#FFE66D"),  # Ball - Yellow
                        sv.Color.from_hex("#95A5A6"),  # Unknown - Gray
                    ]
                )

                corner_annotator = sv.BoxCornerAnnotator(
                    thickness=2, corner_length=15, color=TEAM_COLORS
                )

                label_annotator = sv.LabelAnnotator(
                    text_scale=0.5,
                    text_thickness=1,
                    text_color=sv.Color.from_hex("#000000"),
                    text_padding=5,
                    color=TEAM_COLORS,
                )

                with sv.VideoSink(
                    target_path=str(final_video_file), video_info=video_info
                ) as sink:
                    for frame, detections in zip(render_generator, frames):
                        annotated = frame.copy()

                        labels = []
                        color_indices = []

                        for i in range(len(detections)):
                            class_id = detections.class_id[i]
                            conf = (
                                detections.confidence[i]
                                if detections.confidence is not None
                                else 0.0
                            )

                            if class_id == ClassId.BALL.value:
                                labels.append(f"Ball {conf:.2f}")
                                color_indices.append(2)
                            else:
                                tracker_id = (
                                    detections.tracker_id[i]
                                    if detections.tracker_id is not None
                                    else "?"
                                )
                                team_id = int(detections.data.get("team_id", [-1])[i])
                                labels.append(f"T{team_id} #{tracker_id} {conf:.2f}")
                                color_indices.append(
                                    team_id if team_id in [0, 1] else 3
                                )

                        # Create a proper copy with new class_id array
                        viz_detections = detections[np.arange(len(detections))]
                        viz_detections.class_id = np.array(color_indices)

                        annotated = corner_annotator.annotate(
                            scene=annotated, detections=viz_detections
                        )
                        annotated = label_annotator.annotate(
                            scene=annotated, detections=viz_detections, labels=labels
                        )

                        sink.write_frame(frame=annotated)

        return Frames(frames)
