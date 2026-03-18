from dataclasses import dataclass
import json
import logging
from scipy.stats import gaussian_kde
import numpy as np
from pathlib import Path
from typing import Literal, Any
import supervision as sv

from wunderscout.core import Frames
from wunderscout.types import SaveResult

logger = logging.getLogger(__name__)

HeatmapKey = Literal["histogram", "kde"]


@dataclass
class Heatmap:
    data: dict[HeatmapKey, Any]
    identifier: int
    prefix: Literal["player", "team"]

    def save(
        self,
        output_path: str | Path,
        heatmap_type: Literal["histogram", "kde", "both"] = "both",
    ) -> SaveResult:
        """Save heatmap data to JSON file."""
        path_obj = Path(output_path)
        path_obj.mkdir(parents=True, exist_ok=True)
        types_to_save: list[HeatmapKey] = (
            list(self.data.keys()) if heatmap_type == "both" else [heatmap_type]
        )

        save_result = SaveResult()

        for t in types_to_save:
            path = path_obj / f"{self.prefix}_{self.identifier}_{t}.json"
            try:
                with open(path, "w") as f:
                    json.dump(self.data[t], f)
                save_result.successful_paths.append(path)
            except KeyError as e:
                error_msg = f"Key {e} not found in heatmap data"
                logger.warning(f"Error: {error_msg}")
                save_result.errors.append(str(error_msg))
                save_result.failed_paths.append(path)
            except TypeError as e:
                error_msg = f"Data serialization error for {t}: {e}"
                logger.warning(f"Error: {error_msg}")
                save_result.errors.append(str(error_msg))
                save_result.failed_paths.append(path)
            except Exception as e:
                error_msg = f"Unexpected error saving {t}: {e}"
                logger.warning(f"Error: {error_msg}")
                save_result.errors.append(str(error_msg))
                save_result.failed_paths.append(path)

        return save_result


class HeatmapGenerator:
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        histogram_bins: tuple[int, int] = (50, 34),
        kde_grid_size: tuple[int, int] = (100, 68),
    ):
        """
        Initialize heatmap generator with pitch dimensions and resolution.

        Args:
            pitch_length: Length of pitch in meters (default 105m)
            pitch_width: Width of pitch in meters (default 68m)
            histogram_bins: (x_bins, y_bins) for histogram heatmap
            kde_grid_size: (x_points, y_points) for KDE grid resolution
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.histogram_bins = histogram_bins
        self.kde_grid_size = kde_grid_size
        self.min_samples_for_kde = 10  # min_samples_for_kde: Minimum number of samples required for KDE. Weird behavior due to bad data, will remove as the model improves.

    def _scale_to_meters(self, positions: np.ndarray) -> np.ndarray:
        """Convert normalized [0, 1] coordinates to meters."""
        scaled = positions.copy()
        scaled[:, 0] *= self.pitch_length
        scaled[:, 1] *= self.pitch_width
        return scaled

    def _has_sufficient_variation(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check if data has sufficient spatial variation for KDE."""
        if len(x) < 2:
            return False

        # Check if all points are identical
        x_range = np.ptp(x)  # peak-to-peak (max - min)
        y_range = np.ptp(y)

        # Need at least some variation in both dimensions
        # Using 1cm as minimum threshold
        return x_range > 0.01 and y_range > 0.01

    def _compute_histogram(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Compute 2D histogram heatmap."""
        heatmap, xedges, yedges = np.histogram2d(
            x,
            y,
            bins=self.histogram_bins,
            range=[[0, self.pitch_length], [0, self.pitch_width]],
        )

        return {
            "xedges": xedges.tolist(),
            "yedges": yedges.tolist(),
            "values": heatmap.T.tolist(),
        }

    def _compute_kde(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Compute KDE smoothed density field.

        Returns dict with:
            - x: 1D list of x coordinates
            - y: 1D list of y coordinates
            - values: 2D list where values[i][j] = density at [x[j], y[i]]
        """
        # Add small jitter to prevent perfect collinearity
        # This helps with edge cases where points are nearly identical
        jitter_amount = 0.01  # 1cm jitter
        x_jittered = x + np.random.normal(0, jitter_amount, size=x.shape)
        y_jittered = y + np.random.normal(0, jitter_amount, size=y.shape)

        values = np.vstack([x_jittered, y_jittered])
        kde = gaussian_kde(values)

        # Create coordinate grids
        x_coords = np.linspace(0, self.pitch_length, self.kde_grid_size[0])
        y_coords = np.linspace(0, self.pitch_width, self.kde_grid_size[1])
        X, Y = np.meshgrid(x_coords, y_coords)

        # Evaluate KDE on grid
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)

        return {
            "x": x_coords.tolist(),
            "y": y_coords.tolist(),
            "values": Z.tolist(),
        }

    def team(
        self,
        frames: Frames,
        team_id: int,
        heatmap_type: Literal["histogram", "kde", "both"] = "both",
    ) -> Heatmap:
        """
        Generate aggregated heatmap for entire team.

        Args:
            frames: Frames object
            team_id: Team ID (0 or 1)
            heatmap_type: "histogram", "kde", or "both"
        """
        all_frames = sv.Detections.merge(frames.detections)
        player_ids = all_frames[
            (all_frames.class_id == 2) & (all_frames.data["team_id"] == team_id)
        ].tracker_id

        # Collect all positions from all players
        all_positions = all_frames[np.isin(all_frames.tracker_id, player_ids)].data[
            "pitch_coordinates"
        ]

        positions = np.array(all_positions)
        positions_meters = self._scale_to_meters(positions)
        x, y = positions_meters[:, 0], positions_meters[:, 1]

        output: dict[HeatmapKey, Any] = {}

        # Histogram (always attempt)
        if heatmap_type in ["histogram", "both"]:
            try:
                histogram_result = self._compute_histogram(x, y)
                output["histogram"] = histogram_result
            except Exception as e:
                logger.warning(
                    f"Warning: Team histogram failed for team {team_id}: {e}"
                )
                # Don't include histogram key at all if it fails

        # KDE (with quality checks)
        if heatmap_type in ["kde", "both"]:
            if len(all_positions) < self.min_samples_for_kde:
                logger.debug(
                    f"Info: Team {team_id} has only {len(all_positions)} samples. "
                    f"Skipping KDE."
                )
                # Don't include kde key at all
            elif not self._has_sufficient_variation(x, y):
                logger.debug(
                    f"Info: Team {team_id} has insufficient variation. Skipping KDE."
                )
                # Don't include kde key at all
            else:
                try:
                    kde_result = self._compute_kde(x, y)
                    output["kde"] = kde_result
                except Exception as e:
                    logger.warning(f"Warning: KDE failed for team {team_id}: {e}")
                    # Don't include kde key at all if it fails

        return Heatmap(output, team_id, "team")

    def player(
        self,
        frames: Frames,
        player_id: int,
        heatmap_type: Literal["histogram", "kde", "both"] = "both",
    ) -> Heatmap:
        """
        Generate heatmap for a single player.

        Args:
            frames: Frames object
            player_id: Player tracker ID
            heatmap_type: "histogram", "kde", or "both"

        Returns:
            Dictionary with heatmap data in format ready for JSON export
        """
        merged_frames = sv.Detections.merge(frames.detections)
        positions = merged_frames[merged_frames.tracker_id == player_id].data[
            "pitch_coordinates"
        ]
        # Separate x,y coordinates and flattens positions
        coordinates = np.array(positions)
        coordinates_meters = self._scale_to_meters(coordinates)

        x, y = coordinates_meters[:, 0], coordinates_meters[:, 1]

        output: dict[HeatmapKey, Any] = {}

        # Always try histogram (works with any amount of data)
        if heatmap_type in ["histogram", "both"]:
            try:
                histogram_result = self._compute_histogram(x, y)
                output["histogram"] = histogram_result
            except Exception as e:
                logger.warning(f"Warning: Histogram failed for player {player_id}: {e}")
                # Don't include histogram key at all if it fails

        # Only attempt KDE if we have enough quality data
        if heatmap_type in ["kde", "both"]:
            if len(positions) < self.min_samples_for_kde:
                logger.warning(
                    f"Info: Player {player_id} has only {len(positions)} samples "
                    f"(minimum {self.min_samples_for_kde} required for KDE). "
                    f"Skipping KDE, histogram only."
                )
                # Don't include kde key at all
            elif not self._has_sufficient_variation(x, y):
                logger.warning(
                    f"Info: Player {player_id} has insufficient spatial variation "
                    f"for KDE. Skipping KDE, histogram only."
                )
                # Don't include kde key at all
            else:
                try:
                    kde_result = self._compute_kde(x, y)
                    output["kde"] = kde_result
                except Exception as e:
                    logger.warning(f"Warning: KDE failed for player {player_id}: {e}")
                    # Don't include kde key at all if it fails

        return Heatmap(output, player_id, "player")
