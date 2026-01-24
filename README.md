# wunderscout

A Python library for extracting player and ball tracking data from soccer match footage using YOLO, Siglip embeddings, and homography.

## Features

- **Detection & Tracking**: Uses YOLO for player/ball/pitch-keypoint detection and ByteTrack for temporal consistency.
- **Automated Team Clustering**: Groups players into teams using Siglip vision transformer embeddings and K-Means clustering via UMAP dimensionality reduction.
- **Pitch Mapping**: Transforms 2D image coordinates to a normalized 0-1 coordinate system using pitch keypoint homography.
- **Goalkeeper Attribution**: Assigns goalkeepers to teams based on proximity to team centroids.
- **Data Export**: Generates Team1 and Team2 CSV files containing frame-by-frame XY coordinates.
- **Heatmap Generation**: Creates histogram and KDE-based spatial density maps for individual players and teams.

## Installation

```bash
pip install wunderscout
```

## Quick Start

```python
from wunderscout import Detector, DataExporter, HeatmapGenerator

# Initialize with paths to trained YOLO weights
detector = Detector(
    player_weights="players.pt",
    field_weights="pitch.pt"
)

# Run processing
result = detector.run(
    video_path="input_match.mp4",
    output_video_path="output_match.mp4"
)

# Export tracking data to CSV
exporter = DataExporter()
exporter.save_csvs(result, "output/tracking.csv")

# Generate heatmaps
heatmap_gen = HeatmapGenerator()
player_heatmap = heatmap_gen.generate_player_heatmap(result, player_id=5, method="both")
heatmap_gen.save_heatmap(player_heatmap, "output/player_5.json")
```

## Core Components

### Detector

Main pipeline coordinator that orchestrates detection, tracking, and classification.

**Parameters:**
- `player_weights` (str): Path to YOLO player detection model
- `field_weights` (str): Path to YOLO field keypoint detection model

**Methods:**
- `run(video_path, output_video_path=None)`: Process video and return `TrackingResult`

### VisionEngine

Handles YOLO detection and SiGLIP embedding extraction.

**Key Methods:**
- `detect_players(frame, conf=0.3)`: Detect players, goalkeepers, referees, and ball
- `detect_field(frame, conf=0.3)`: Detect pitch keypoints for homography
- `get_embeddings(pil_crops, batch_size=32)`: Extract visual embeddings for team classification
- `get_calibration_crops(video_path, stride=30)`: Sample player crops for calibration

**Detection Class IDs:**
- 0: Ball
- 1: Goalkeeper
- 2: Player
- 3: Referee

### TeamClassifier

Clusters players into teams using UMAP + K-Means with temporal smoothing.

**Methods:**
- `fit(embeddings)`: Train clustering model on calibration embeddings
- `get_consensus_team(tracker_id, embedding)`: Assign team ID with 50-frame rolling consensus
- `resolve_goalkeepers_team_id(players, goalkeepers)`: Assign goalkeepers by centroid proximity
- `get_final_assignments()`: Get final team assignments (dict: tracker_id → team_id)

### PitchMapper

Projects pixel coordinates to normalized pitch coordinates [0, 1] × [0, 1] using homography.

**Methods:**
- `get_matrix(keypoints_xy, keypoints_conf)`: Compute homography from detected pitch keypoints (requires ≥4 keypoints with conf > 0.5)
- `transform(points, H=None)`: Transform points to normalized pitch coordinates

### HeatmapGenerator

Generates spatial density maps using histogram binning and/or Gaussian KDE.

**Parameters:**
- `pitch_length` (float): Pitch length in meters (default: 105.0)
- `pitch_width` (float): Pitch width in meters (default: 68.0)
- `histogram_bins` (tuple): Histogram resolution (default: (50, 34))
- `kde_grid_size` (tuple): KDE grid resolution (default: (100, 68))
- `min_samples_for_kde` (int): Minimum samples for KDE (default: 10)

**Methods:**
- `generate_player_heatmap(result, player_id, method="both")`: Generate heatmap for one player
- `generate_team_heatmap(result, team, method="both")`: Generate aggregated team heatmap
- `generate_all_players_heatmaps(result, method="both")`: Generate heatmaps for all players
- `save_heatmap(heatmap_data, output_path, pretty=False)`: Save heatmap to JSON

### DataExporter

Exports tracking data to CSV format.

**Methods:**
- `save_csvs(result, output_path)`: Export tracking data (creates `{base_name}_Team1.csv` and `{base_name}_Team2.csv`)

### TrackingResult

Data class containing tracking results.

**Attributes:**
- `frames` (dict): Frame-by-frame tracking data `{frame_idx: {"players": {player_id: (x, y)}, "ball": (x, y) or None}}`
- `team_assignments` (dict): Player ID → Team ID mapping
- `total_frames` (int): Total number of processed frames
- `fps` (float): Video frame rate

**Methods:**
- `get_team_players(team)`: Get player IDs for team 0 or 1
- `get_all_player_ids()`: Get all tracked player IDs
- `get_player_trajectory(player_id)`: Get position history for one player
- `get_ball_trajectory()`: Get ball position history

## Training Custom Models

```python
from wunderscout.vision import ScoutingTrainer

trainer = ScoutingTrainer(api_key="your_roboflow_api_key")

# Train player detection model
trainer.train_players(
    workspace="soccer-analytics",
    project="player-detection",
    version=1,
    epochs=300,
    output_dir="runs/training/player"
)

# Train field keypoint model
trainer.train_field(
    workspace="soccer-analytics",
    project="field-keypoints",
    version=1,
    epochs=300,
    output_dir="runs/training/field"
)
```

## Output Formats

### CSV Export

Two CSV files (one per team) with structure:

```csv
,,,Team1,Team1,Team1,Team1,,
,,,3,3,7,7,,
Period,Frame,Time [s],Player3_X,Player3_Y,Player7_X,Player7_Y,Ball_X,Ball_Y
1,0,0.00,0.234,0.567,0.789,0.345,0.500,0.500
1,1,0.04,0.235,0.568,NaN,NaN,0.501,0.499
```

**Coordinate System:**
- Range: [0, 1] × [0, 1]
- (0, 0) = Top-left corner of pitch
- (1, 1) = Bottom-right corner of pitch
- Missing data: "NaN"

### Heatmap JSON

```json
{
  "player_id": 5,
  "sample_count": 1500,
  "histogram": {
    "xedges": [0.0, 2.1, 4.2, ..., 105.0],
    "yedges": [0.0, 2.0, 4.0, ..., 68.0],
    "values": [[count_00, count_01, ...], [count_10, count_11, ...], ...]
  },
  "kde": {
    "x": [0.0, 1.05, 2.1, ..., 105.0],
    "y": [0.0, 1.0, 2.0, ..., 68.0],
    "values": [[density_00, density_01, ...], [density_10, density_11, ...], ...]
  }
}
```

## Pitch Coordinate System

Normalized coordinate system mapping to standard football pitch (105m × 68m).

**Coordinate Range:**
- X-axis: 0.0 (left goal line) → 1.0 (right goal line)
- Y-axis: 0.0 (top sideline) → 1.0 (bottom sideline)

**Key Landmarks:**

| Landmark | X | Y | Description |
|----------|---|---|-------------|
| Left goal (top post) | 0.000 | 0.365 | Top edge of left goal |
| Left goal (bottom post) | 0.000 | 0.635 | Bottom edge of left goal |
| Left penalty spot | 0.105 | 0.500 | 11m from goal line |
| Center circle | 0.500 | 0.500 | Pitch center |
| Right penalty spot | 0.895 | 0.500 | 11m from goal line |
| Right goal (top post) | 1.000 | 0.365 | Top edge of right goal |
| Right goal (bottom post) | 1.000 | 0.635 | Bottom edge of right goal |

## Dependencies

- `ultralytics`
- `supervision`
- `transformers`
- `umap-learn`
- `scikit-learn`
- `opencv-python`
- `scipy`
- `more-itertools`
- `torch`
- `numpy`
- `roboflow`

## License

MIT
