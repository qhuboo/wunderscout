import csv
from pathlib import Path
from dataclasses import dataclass, field
import logging
import supervision as sv
import numpy as np
from torch import NoneType

from wunderscout.types import SaveResult

from .models import Models
from .geometry import PitchMapper
from .teams import TeamClassifier

logger = logging.getLogger(__name__)

# Class ID constants
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


@dataclass
class Frames:
    detections: list[sv.Detections]
    _player_team_map: dict[int, int] = field(init=False)

    def __post_init__(self):
        self._player_team_map = self._build_player_team_map()

    def _build_player_team_map(self):
        merged = sv.Detections.merge(self.detections)
        if merged.tracker_id is None:
            return {}

        all_players = merged[merged.class_id == PLAYER_ID]

        player_team_map = {
            int(tracker_id): int(team_id)
            for tracker_id, team_id in zip(
                all_players.tracker_id, all_players.data["team_id"]
            )
        }

        return player_team_map

    def get_all_player_ids(self):
        return list(self._player_team_map.keys())

    def get_all_team_ids(self):
        return list(set(self._player_team_map.values()))

    def get_team_for_player(self, player_id: int):
        return self._player_team_map[player_id]

    def save_csvs(self, output_path: str | Path) -> SaveResult:
        """Export tracking data to CSV files (one per team)."""
        path_obj = Path(output_path)
        path_obj.mkdir(parents=True, exist_ok=True)

        team_ids = list(self.get_all_team_ids())
        team_players = {
            team_id: [
                pid for pid, tid in self._player_team_map.items() if tid == team_id
            ]
            for team_id in team_ids
        }

        def write_file(path, team_name, player_ids):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)

                # Row 1: Team names
                writer.writerow(
                    ["", "", ""]
                    + [item for _ in player_ids for item in (team_name, "")]
                    + ["", ""]
                )
                writer.writerow(
                    ["", "", ""]
                    + [item for pid in player_ids for item in (str(pid), "")]
                    + ["", ""]
                )
                writer.writerow(
                    ["Period", "Frame", "Time [s]"]
                    + [item for pid in player_ids for item in (f"Player{pid}", "")]
                    + ["Ball", ""]
                )

                for f_idx, detection in enumerate(self.detections):
                    # period, frame, time
                    row = [1, f_idx, ""]

                    # Get player positions
                    for player_id in player_ids:
                        if detection.tracker_id is not None:
                            mask = detection.tracker_id == player_id
                            if mask.any():
                                idx = np.where(mask)[0][0]
                                coords = detection.data["pitch_coordinates"][idx]
                                row.extend([f"{coords[0]:.5f}", f"{coords[1]:.5f}"])
                            else:
                                row.extend(["NaN", "NaN"])
                        else:
                            row.extend(["NaN", "NaN"])

                    # Get ball position
                    ball_mask = detection.class_id == BALL_ID
                    if ball_mask.any():
                        idx = np.where(ball_mask)[0][0]
                        coords = detection.data["pitch_coordinates"][idx]
                        row.extend([f"{coords[0]:.5f}", f"{coords[1]:.5f}"])
                    else:
                        row.extend(["NaN", "NaN"])

                    writer.writerow(row)

        save_result = SaveResult()

        for team_id in team_ids:
            path = path_obj / f"team_{team_id}.csv"
            try:
                write_file(path, f"Team{team_id}", team_players[team_id])
                save_result.successful_paths.append(path)
            except Exception as e:
                error_msg = f"Unexpected error saving Team{team_id}.csv: {e}"
                logger.warning(f"Error: {error_msg}")
                save_result.errors.append(str(error_msg))
                save_result.failed_paths.append(path)

        return save_result


class Detector:
    def __init__(self, models: Models):
        self.models = models
        self.mapper = PitchMapper()
        self.classifier = TeamClassifier()

    def run(self, video_path, save_video_path: str | Path | None = None) -> Frames:
        crops = self.models.get_calibration_crops(video_path, class_id=PLAYER_ID)
        if len(crops) > 0:
            embeddings = self.models.get_embeddings(crops)
            self.classifier.fit(embeddings)
        else:
            logger.warning("WARNING: No player crops found for calibration.")

        tracker = sv.ByteTrack()
        frames = []

        # 3. Main Processing Loop
        logger.debug(f"Starting processing: {video_path}")
        frame_generator = sv.get_video_frames_generator(video_path)
        for frame_idx, frame in enumerate(frame_generator):
            logger.debug(f"Processing frame {frame_idx}")
            # --- A. DETECTION ---
            all_dets = self.models.detect_players(frame)
            f_res = self.models.detect_field(frame)

            # --- B. FIELD HOMOGRAPHY ---
            H = None
            if f_res.keypoints is not None and len(f_res.keypoints.xy) > 0:
                H = self.mapper.get_matrix(
                    f_res.keypoints.xy[0].cpu().numpy(),
                    f_res.keypoints.conf[0].cpu().numpy(),
                )
            else:
                H = self.mapper.last_h

            logger.debug(f"H: {H}")

            # --- C. SEPARATE BALL & OTHERS ---
            ball_detections = all_dets[all_dets.class_id == BALL_ID]
            other_detections = all_dets[all_dets.class_id != BALL_ID]
            other_detections = other_detections.with_nms(threshold=0.5)

            # --- D. TRACKING ---
            tracked_objects = tracker.update_with_detections(other_detections)

            # Split tracked objects
            tracked_players = tracked_objects[tracked_objects.class_id == PLAYER_ID]
            tracked_gks = tracked_objects[tracked_objects.class_id == GOALKEEPER_ID]

            # Pad ball_detections with tracker_ids to avoid error on merge
            ball_detections.tracker_id = np.array(
                [-1] * len(ball_detections), dtype=int
            )

            # --- E. TEAM CLASSIFICATION ---

            # 1. Players
            if len(tracked_players) > 0:
                p_crops = [sv.crop_image(frame, xyxy) for xyxy in tracked_players.xyxy]
                p_pil = [sv.cv2_to_pillow(c) for c in p_crops]
                p_embeddings = self.models.get_embeddings(p_pil)

                final_team_ids = self.classifier.get_consensus_teams(
                    tracked_players.tracker_id, p_embeddings
                )

                tracked_players.data["team_id"] = np.array(final_team_ids)

            else:
                tracked_players.data["team_id"] = np.array([], dtype=int)

            # 2. Goalkeepers
            if len(tracked_gks) > 0 and len(tracked_players) > 0:
                tracked_gks.data["team_id"] = (
                    self.classifier.resolve_goalkeepers_team_id(
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
                    data_targets.data["pitch_coordinates"] = self.mapper.transform(
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
                    ball_detections.data["pitch_coordinates"] = self.mapper.transform(
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
            if save_video_path is not None:
                out_dir = Path(save_video_path)
                out_dir.mkdir(parents=True, exist_ok=True)

                orig_path = Path(video_path)
                new_filename = f"{orig_path.stem}_annotated{orig_path.suffix}"
                final_video_file = out_dir / new_filename

                logger.info(f"Generating annotated video at {final_video_file}...")

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

                            if class_id == BALL_ID:
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
