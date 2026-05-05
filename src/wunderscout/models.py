import logging
import torch
from ultralytics import YOLO
import supervision as sv
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import numpy as np

logger = logging.getLogger(__name__)


class Models:
    def __init__(self, player_weights, field_weights, siglip_path=None):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._player_model = YOLO(player_weights)
        self._player_model.to(self._device)
        self._field_model = YOLO(field_weights)
        self._field_model.to(self._device)

        siglip_path = siglip_path or "google/siglip-base-patch16-224"
        self._siglip_model = SiglipVisionModel.from_pretrained(siglip_path).to(
            self._device
        )
        self._siglip_processor = AutoProcessor.from_pretrained(siglip_path)

    # TODO: Add detailed docstring, logging, and better error handling
    def _get_calibration_crops(self, video_path, class_id, stride=30):
        """
        Extract crops of detected objects for calibration.

        Args:
            video_path: Path to video file.
            class_id: Object class to detect.
            stride: Stride for detection

        Returns:
            List of PIL Image crops.

        Note:
            Returns empty list if no detections found.
        """
        logger.debug(f"Extracting calibration crops for class_id: {class_id}.")
        frame_generator = sv.get_video_frames_generator(
            source_path=video_path, stride=stride
        )

        crops = []
        frame_count = 0
        total_detections = 0
        for frame in frame_generator:
            frame_count += 1
            logger.debug(f"Processing frame {frame_count}, shape: {frame.shape}.")

            detections = self._detect_players(frame)
            total_detections += len(detections)

            logger.debug(
                f"Frame {frame_count}: {len(detections)} total detections, "
                f"class_ids: {detections.class_id.tolist() if len(detections) > 0 else []}"
            )
            players = detections[detections.class_id == class_id]
            logger.debug(
                f"Frame {frame_count}: {len(players)} matching class_id={class_id}"
            )
            frame_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]

            # Filter invalid crops
            valid_crops = [c for c in frame_crops if c is not None and c.size > 0]
            crops += [sv.cv2_to_pillow(c) for c in valid_crops]

        logger.info(
            f"Calibration complete: processed {frame_count} frames, "
            f"{total_detections} total detections, "
            f"{len(crops)} crops extracted for class_id={class_id}"
        )
        return crops

    # TODO: Add detailed docstring, logging, and better error handling
    def _get_embeddings(self, pil_crops, batch_size=32):
        batches = chunked(pil_crops, batch_size)
        data_list = []

        with torch.no_grad():
            for batch in batches:
                inputs = self._siglip_processor(images=batch, return_tensors="pt").to(
                    self._device
                )
                outputs = self._siglip_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data_list.append(embeddings)

        return np.concatenate(data_list) if data_list else np.array([])

    # TODO: Add detailed docstring, logging, and better error handling
    def _detect_players(self, frame, conf=0.0) -> sv.Detections:
        result = self._player_model.predict(
            frame, conf=conf, verbose=False, device=self._device
        )[0]
        return sv.Detections.from_ultralytics(result)

    # TODO: Add detailed docstring, logging, and better error handling
    def _detect_field(self, frame, conf=0.0) -> sv.Detections:
        result = self._field_model.predict(
            frame, conf=conf, verbose=False, device=self._device
        )[0]
        return result
