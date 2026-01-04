# wunderscout

A Python library for extracting player and ball tracking data from soccer match footage using YOLO, Siglip embeddings, and homography.

## Features

- **Detection & Tracking**: Uses YOLO for player/ball/pitch-keypoint detection and ByteTrack for temporal consistency.
- **Automated Team Clustering**: Groups players into teams using Siglip vision transformer embeddings and K-Means clustering via UMAP dimensionality reduction.
- **Pitch Mapping**: Transforms 2D image coordinates to a normalized 0-1 coordinate system using pitch keypoint homography.
- **Goalkeeper Attribution**: Assigns goalkeepers to teams based on proximity to team centroids.
- **Data Export**: Generates Home and Away CSV files containing frame-by-frame XY coordinates.

## Installation

```bash
uv add wunderscout
```

## Usage

The `ScoutingPipeline` class manages calibration, tracking, and data export.

```python
from wunderscout import ScoutingPipeline

# Initialize with paths to trained YOLO weights
pipeline = ScoutingPipeline(
    player_weights="players.pt",
    field_weights="pitch.pt"
)

# Run processing
pipeline.run(
    video_path="input_match.mp4",
    output_video_path="ouput_match.mp4"
)
```

## Internal Components

- **VisionEngine**: Manages YOLO models and generates Siglip embeddings for player crops.
- **PitchMapper**: Computes homography matrices based on a 32-point pitch configuration. Handles RANSAC-based perspective transforms.
- **TeamClassifier**: Performs unsupervised clustering on player embeddings. Uses a rolling consensus buffer to stabilize team assignments across frames.
- **DataExporter**: Formats tracking results into CSV files with frame indices and normalized pitch coordinates.

## Dependencies

- `ultralytics`
- `supervision`
- `transformers`
- `umap-learn`
- `scikit-learn`
- `opencv-python`

## License

MIT
