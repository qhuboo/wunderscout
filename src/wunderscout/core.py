import cv2
import supervision as sv
import numpy as np
from pathlib import Path
from .vision import VisionEngine
from .geometry import PitchMapper
from .teams import TeamClassifier
from .data import TrackingResult


class Detector:
    def __init__(self, player_weights, field_weights):
        self.engine = VisionEngine(player_weights, field_weights)
        self.mapper = PitchMapper()
        self.classifier = TeamClassifier()

    def run(self, video_path, output_video_path=None):
        # 1. Warm-up (Calibration)
        print("Calibrating teams...")
        crops = self.engine.get_calibration_crops(video_path)
        if len(crops) > 0:
            embeddings = self.engine.get_embeddings(crops)
            self.classifier.fit(embeddings)
        else:
            print("WARNING: No player crops found for calibration.")

        # 2. Setup Video I/O
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_video_path:
            output_path_obj = Path(output_video_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            out = cv2.VideoWriter(
                output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not out.isOpened():
                print(f"ERROR: Could not create video file at {output_video_path}")
                out = None

        tracker = sv.ByteTrack()
        tracking_results = {}

        # ID Constants
        BALL_ID = 0
        GOALKEEPER_ID = 1
        PLAYER_ID = 2
        REFEREE_ID = 3

        # 3. Main Processing Loop
        print(f"Starting processing: {video_path}")
        frame_generator = sv.get_video_frames_generator(video_path)

        frame_idx = -1
        for frame_idx, frame in enumerate(frame_generator):
            print(f"Processing frame {frame_idx}")

            # --- A. DETECTION ---
            all_dets = self.engine.detect_players(frame)
            f_res = self.engine.detect_field(frame)

            # --- B. FIELD HOMOGRAPHY ---
            H = None
            if f_res.keypoints is not None and len(f_res.keypoints.xy) > 0:
                H = self.mapper.get_matrix(
                    f_res.keypoints.xy[0].cpu().numpy(),
                    f_res.keypoints.conf[0].cpu().numpy(),
                )
            else:
                H = self.mapper.last_h

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
            tracked_refs = tracked_objects[tracked_objects.class_id == REFEREE_ID]

            # --- E. TEAM CLASSIFICATION ---

            # 1. Players
            if len(tracked_players) > 0:
                p_crops = [sv.crop_image(frame, xyxy) for xyxy in tracked_players.xyxy]
                p_pil = [sv.cv2_to_pillow(c) for c in p_crops]
                p_embeddings = self.engine.get_embeddings(p_pil)

                final_team_ids = []
                for i, tid in enumerate(tracked_players.tracker_id):
                    team_id = self.classifier.get_consensus_team(tid, p_embeddings[i])
                    final_team_ids.append(team_id)

                tracked_players.class_id = np.array(final_team_ids)

            # 2. Goalkeepers
            if len(tracked_gks) > 0 and len(tracked_players) > 0:
                tracked_gks.class_id = self.classifier.resolve_goalkeepers_team_id(
                    tracked_players, tracked_gks
                )

            # 3. Referees (Shift ID 3 -> 2)
            if len(tracked_refs) > 0:
                tracked_refs.class_id -= 1

            # --- F. DATA STORAGE ---
            tracking_results[frame_idx] = {"players": {}, "ball": None}
            data_targets = sv.Detections.merge([tracked_players, tracked_gks])

            if H is not None:
                if len(data_targets) > 0:
                    feet_coords = data_targets.get_anchors_coordinates(
                        sv.Position.BOTTOM_CENTER
                    )
                    transformed_feet = self.mapper.transform(feet_coords, H)

                    for i, tid in enumerate(data_targets.tracker_id):
                        px, py = transformed_feet[i]
                        tracking_results[frame_idx]["players"][tid] = (
                            max(0.0, min(1.0, px)),
                            max(0.0, min(1.0, py)),
                        )

                if len(ball_detections) > 0:
                    ball_coords = ball_detections.get_anchors_coordinates(
                        sv.Position.CENTER
                    )
                    transformed_ball = self.mapper.transform([ball_coords[0]], H)
                    bx, by = transformed_ball[0]
                    tracking_results[frame_idx]["ball"] = (
                        max(0.0, min(1.0, bx)),
                        max(0.0, min(1.0, by)),
                    )

            # --- G. DRAW & WRITE VIDEO ---
            if out:
                all_tracked = sv.Detections.merge(
                    [tracked_players, tracked_gks, tracked_refs]
                )
                annotated_frame = self.engine.draw_annotations(
                    frame, all_tracked, ball_detections
                )
                out.write(annotated_frame)

        # 4. Cleanup
        if out:
            out.release()
            print(f"Video saved to {output_video_path}")
        cap.release()

        # 5. Return data
        return TrackingResult(
            frames=tracking_results,
            team_assignments=self.classifier.get_final_assignments(),
            total_frames=frame_idx + 1,
            fps=fps,
        )
