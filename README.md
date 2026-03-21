# WunderScout

> ⚠️ **EXPERIMENTAL PROJECT** ⚠️
>
> This is an experimental computer vision project for soccer/football video analysis. It is **NOT** production-ready and should be used for research, experimentation, and educational purposes only. The API may change without notice, and results may not be accurate or reliable for professional or commercial applications.

WunderScout is a Python library for automated soccer/football video analysis using computer vision and machine learning. It provides tools for player detection, tracking, team classification, and positional heatmap generation from match footage.

## Features

- **Player & Ball Detection**: Automatic detection of players, goalkeepers, and ball in video frames
- **Multi-Object Tracking**: Track players across frames with unique IDs
- **Team Classification**: Automatic team assignment using visual embeddings
- **Pitch Mapping**: Transform image coordinates to real-world pitch coordinates
- **Heatmap Generation**: Create spatial heatmaps for individual players or entire teams
- **CSV Export**: Export tracking data in standardized CSV format
- **Video Annotation**: Generate annotated videos with bounding boxes and labels

## Installation

```bash
pip install wunderscout
```

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for performance)
- Model weights for player detection and field keypoint detection

## Quick Start

```python
import wunderscout

# Enable logging (optional)
wunderscout.set_stream_logger('wunderscout', level=logging.INFO)

# Initialize models
models = wunderscout.Models(
    player_weights='path/to/player_model.pt',
    field_weights='path/to/field_model.pt'
)

# Create detector
detector = wunderscout.Detector(models)

# Process video
frames = detector.run(
    video_path='match.mp4',
    output_dir='./output'  # Optional: save annotated video
)

# Export tracking data
frames.save_csvs('./tracking_data')

# Generate heatmaps
heatmap_gen = wunderscout.HeatmapGenerator()
player_heatmap = heatmap_gen.player(frames, player_id=5)
player_heatmap.save('./heatmaps', heatmap_type='both')
```

## API Reference

### Core Classes

#### `Models`

Manages the machine learning models for detection and classification.

```python
Models(
    player_weights: str,
    field_weights: str,
    siglip_path: str = None
)
```

**Parameters:**

- `player_weights`: Path to YOLO weights file for player/ball detection
- `field_weights`: Path to YOLO weights file for field keypoint detection
- `siglip_path`: Path or identifier for SiglipVisionModel (default: "google/siglip-base-patch16-224")

---

#### `Detector`

Main class for running detection and tracking on video footage.

```python
Detector(models: Models)
```

**Methods:**

##### `run(video_path, output_dir=None) -> Frames`

Process a video file and return tracking results.

**Parameters:**

- `video_path` (str | Path): Path to input video file
- `output_dir` (str | Path | None): Optional directory to save annotated video

**Returns:**

- `Frames`: Object containing detection results for all frames

**Raises:**

- `FileNotFoundError`: If video file not found
- `ValueError`: If video format is invalid or output_dir is a file
- `OSError`: If video data cannot be read
- `RuntimeError`: If calibration fails

**Example:**

```python
detector = Detector(models)
frames = detector.run('match.mp4', output_dir='./annotated')
```

---

#### `Frames`

Container for detection results across all video frames.

**Methods:**

##### `get_all_player_ids() -> list[int]`

Get tracker IDs for all detected players.

**Returns:**

- List of player tracker IDs

##### `get_all_team_ids() -> list[int]`

Get all unique team IDs.

**Returns:**

- List of team IDs (typically [0, 1])

##### `get_team_for_player(player_id: int) -> int`

Get team ID for a specific player.

**Parameters:**

- `player_id`: Player tracker ID

**Returns:**

- Team ID (0 or 1)

##### `save_csvs(output_path: str | Path) -> SaveResult`

Export tracking data to CSV files (one per team).

**Parameters:**

- `output_path`: Directory path for output CSV files

**Returns:**

- `SaveResult`: Object containing success/failure information

**CSV Format:**

- Row 1: Team names
- Row 2: Player IDs
- Row 3: Column headers (Period, Frame, Time, Player coordinates, Ball coordinates)
- Subsequent rows: Frame-by-frame tracking data with pitch coordinates

**Example:**

```python
result = frames.save_csvs('./output')
print(f"Saved: {result.successful_paths}")
print(f"Errors: {result.errors}")
```

---

#### `HeatmapGenerator`

Generate spatial heatmaps for players and teams.

```python
HeatmapGenerator(
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    histogram_bins: tuple[int, int] = (50, 34),
    kde_grid_size: tuple[int, int] = (100, 68)
)
```

**Parameters:**

- `pitch_length`: Pitch length in meters (default: 105.0)
- `pitch_width`: Pitch width in meters (default: 68.0)
- `histogram_bins`: (x_bins, y_bins) for histogram resolution
- `kde_grid_size`: (x_points, y_points) for KDE grid resolution

**Methods:**

##### `player(frames: Frames, player_id: int, heatmap_type='both') -> Heatmap`

Generate heatmap for a single player.

**Parameters:**

- `frames`: Frames object from detector
- `player_id`: Player tracker ID
- `heatmap_type`: One of "histogram", "kde", or "both"

**Returns:**

- `Heatmap`: Heatmap object with data

##### `team(frames: Frames, team_id: int, heatmap_type='both') -> Heatmap`

Generate aggregated heatmap for entire team.

**Parameters:**

- `frames`: Frames object from detector
- `team_id`: Team ID (0 or 1)
- `heatmap_type`: One of "histogram", "kde", or "both"

**Returns:**

- `Heatmap`: Heatmap object with data

**Example:**

```python
gen = HeatmapGenerator(pitch_length=105.0, pitch_width=68.0)

# Player heatmap
player_hm = gen.player(frames, player_id=7, heatmap_type='both')
player_hm.save('./heatmaps')

# Team heatmap
team_hm = gen.team(frames, team_id=0, heatmap_type='kde')
team_hm.save('./heatmaps')
```

---

#### `Heatmap`

Container for heatmap data.

**Attributes:**

- `data`: Dictionary containing heatmap data
- `identifier`: Player or team ID
- `prefix`: Either "player" or "team"

**Methods:**

##### `save(output_path: str | Path, heatmap_type='both') -> SaveResult`

Save heatmap data to JSON file(s).

**Parameters:**

- `output_path`: Directory path for output files
- `heatmap_type`: One of "histogram", "kde", or "both"

**Returns:**

- `SaveResult`: Object containing success/failure information

**Output Format:**

- Histogram: `{prefix}_{identifier}_histogram.json`
- KDE: `{prefix}_{identifier}_kde.json`

**Example:**

```python
heatmap = gen.player(frames, player_id=10)
result = heatmap.save('./output', heatmap_type='both')
```

---

#### `SaveResult`

Result object for save operations.

**Attributes:**

- `successful_paths`: List of successfully saved file paths
- `failed_paths`: List of failed file paths
- `errors`: List of error messages

---

### Utility Functions

#### `set_stream_logger(name='wunderscout', level=logging.DEBUG, format_string=None)`

Configure stream logging for WunderScout.

**Parameters:**

- `name`: Logger name (default: 'wunderscout')
- `level`: Logging level (e.g., `logging.INFO`, `logging.DEBUG`)
- `format_string`: Optional custom format string

**Example:**

```python
import logging
import wunderscout

wunderscout.set_stream_logger('wunderscout', level=logging.INFO)
```

---

## Complete Example

```python
import logging
import wunderscout

# Setup logging
wunderscout.set_stream_logger(name='wunderscout', level=logging.INFO)

# Initialize models
models = wunderscout.Models(
    player_weights='models/player_detection.pt',
    field_weights='models/field_keypoints.pt'
)

# Create detector
detector = wunderscout.Detector(models)

# Process video
frames = detector.run(
    video_path='match.mp4',
    output_dir='./annotated_videos'
)

# Get player and team information
all_players = frames.get_all_player_ids()
all_teams = frames.get_all_team_ids()

print(f"Detected {len(all_players)} players across {len(all_teams)} teams")

# Export tracking data
csv_result = frames.save_csvs('./tracking_data')
print(f"Exported CSV files: {csv_result.successful_paths}")

# Generate heatmaps
heatmap_gen = wunderscout.HeatmapGenerator()

# Individual player heatmaps
for player_id in all_players[:5]:  # First 5 players
    team_id = frames.get_team_for_player(player_id)
    heatmap = heatmap_gen.player(frames, player_id, heatmap_type='both')
    heatmap.save(f'./heatmaps/player_{player_id}')

# Team heatmaps
for team_id in all_teams:
    heatmap = heatmap_gen.team(frames, team_id, heatmap_type='kde')
    heatmap.save(f'./heatmaps/team_{team_id}')

print("Analysis complete!")
```

## Coordinate System

WunderScout uses a normalized pitch coordinate system:

- Origin (0, 0) at top-left corner
- (1, 1) at bottom-right corner
- Default pitch dimensions: 105m × 68m (standard FIFA pitch)

Coordinates are transformed from image space to pitch space using homography matrices computed from detected field keypoints.

## Limitations

- Requires clear visibility of pitch markings for accurate coordinate mapping
- Performance depends on quality of input video and model weights
- Team classification may struggle with similar team colors
- KDE heatmaps require minimum sample size and spatial variation

## License

MIT License

## Acknowledgments

Built with:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Supervision](https://github.com/roboflow/supervision)
- [Transformers](https://github.com/huggingface/transformers)
- [OpenCV](https://opencv.org/)
