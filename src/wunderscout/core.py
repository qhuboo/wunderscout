import logging
import supervision as sv
import numpy as np

from wunderscout import data
from .models import Models
from .geometry import PitchMapper
from .teams import TeamClassifier

logger = logging.getLogger(__name__)

# Class ID constants
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


class Detector:
    def __init__(self, models: Models):
        self.models = models
        self.mapper = PitchMapper()
        self.classifier = TeamClassifier()

    def run(self, video_path):
        crops = self.models.get_calibration_crops(video_path, class_id=PLAYER_ID)
        if len(crops) > 0:
            embeddings = self.models.get_embeddings(crops)
            self.classifier.fit(embeddings)
        else:
            logger.warning("WARNING: No player crops found for calibration.")

        tracker = sv.ByteTrack()
        frames = []

        # 3. Main Processing Loop
        logger.info(f"Starting processing: {video_path}")
        frame_generator = sv.get_video_frames_generator(video_path)
        for frame_idx, frame in enumerate(frame_generator):
            logger.info(f"Processing frame {frame_idx}")
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
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            other_detections = all_dets[all_dets.class_id != BALL_ID]
            other_detections = other_detections.with_nms(threshold=0.5)

            # --- D. TRACKING ---
            tracked_objects = tracker.update_with_detections(other_detections)

            # Split tracked objects
            tracked_players = tracked_objects[tracked_objects.class_id == PLAYER_ID]
            tracked_gks = tracked_objects[tracked_objects.class_id == GOALKEEPER_ID]

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

            frames.append(sv.Detections.merge([data_targets, ball_detections]))

        return frames
