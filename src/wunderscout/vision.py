import torch
from ultralytics import YOLO
import supervision as sv
from transformers import AutoProcessor, SiglipVisionModel
from roboflow import Roboflow
from more_itertools import chunked
import numpy as np


class VisionEngine:
    def __init__(self, player_weights, field_weights, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.player_model = YOLO(player_weights)
        self.field_model = YOLO(field_weights)

        # Siglip for embeddings
        siglip_path = "google/siglip-base-patch16-224"
        self.siglip_model = SiglipVisionModel.from_pretrained(siglip_path).to(
            self.device
        )
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_path)

        # --- Annotators ---
        # Palette: 0=Blue, 1=Pink, 2=Yellow (Referee)
        self.palette = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"])

        self.ellipse_annotator = sv.EllipseAnnotator(
            color=self.palette,
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=self.palette,
            text_color=sv.Color.from_hex("#000000"),
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"), base=25, height=21, outline_thickness=1
        )

    def get_calibration_crops(self, video_path, stride=30):
        PLAYER_ID = 2
        frame_generator = sv.get_video_frames_generator(
            source_path=video_path, stride=stride
        )

        crops = []
        for frame in frame_generator:
            detections = self.detect_players(frame)
            # Filter for players only for calibration
            players = detections[detections.class_id == PLAYER_ID]
            frame_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
            crops += [sv.cv2_to_pillow(c) for c in frame_crops]

        print(f"VisionEngine: Collected {len(crops)} calibration crops.")
        return crops

    def get_embeddings(self, pil_crops, batch_size=32):
        batches = chunked(pil_crops, batch_size)
        data_list = []

        with torch.no_grad():
            for batch in batches:
                inputs = self.siglip_processor(images=batch, return_tensors="pt").to(
                    self.device
                )
                outputs = self.siglip_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data_list.append(embeddings)

        return np.concatenate(data_list) if data_list else np.array([])

    def detect_players(self, frame, conf=0.3):
        result = self.player_model.predict(frame, conf=conf, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def detect_field(self, frame, conf=0.3):
        result = self.field_model.predict(frame, conf=conf, verbose=False)[0]
        return result

    def draw_annotations(self, frame, all_detections, ball_detections):
        annotated_frame = frame.copy()

        # 1. Draw Ball
        annotated_frame = self.triangle_annotator.annotate(
            scene=annotated_frame, detections=ball_detections
        )

        # 2. Draw People (Players, GKs, Refs)
        if len(all_detections) > 0:
            # Ensure class_id is int for color mapping
            all_detections.class_id = all_detections.class_id.astype(int)

            labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

            annotated_frame = self.ellipse_annotator.annotate(
                scene=annotated_frame, detections=all_detections
            )
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=all_detections, labels=labels
            )

        return annotated_frame


class ScoutingTrainer:
    def __init__(self, api_key):
        self.rf = Roboflow(api_key=api_key)

    def train_players(
        self,
        workspace,
        project,
        version,
        epochs=300,
        output_dir="../runs/training/player",
    ):
        project = self.rf.workspace(workspace).project(project)
        dataset = project.version(version).download("yolov11")
        model = YOLO("../data/base_models/yolo11m.pt")

        return model.train(
            data=f"{dataset.location}/data.yaml",
            epochs=epochs,
            imgsz=1280,
            plots=True,
            device=0,
            batch=2,
            project=output_dir,
        )

    def train_field(
        self,
        workspace,
        project,
        version,
        epochs=300,
        output_dir="../runs/training/field",
    ):
        project = self.rf.workspace(workspace).project(project)
        version = project.version(15)
        dataset = version.download("yolov8", location="../data/data_sets/")
        model = YOLO("yolo11m-pose.pt")

        return model.train(
            data=f"{dataset.location}/data.yaml",
            save=True,
            epochs=epochs,
            plots=True,
            imgsz=1080,
            device=0,
            batch=2,
            project=output_dir,
        )
