import numpy as np
import json
from scipy.stats import gaussian_kde
from pathlib import Path
from typing import Literal, Any
from .data import TrackingResult


class HeatmapGenerator:
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        histogram_bins: tuple[int, int] = (50, 34),
        kde_grid_size: tuple[int, int] = (100, 68),
        min_samples_for_kde: int = 10,  # Minimum samples needed for KDE
    ):
        """
        Initialize heatmap generator with pitch dimensions and resolution.

        Args:
            pitch_length: Length of pitch in meters (default 105m)
            pitch_width: Width of pitch in meters (default 68m)
            histogram_bins: (x_bins, y_bins) for histogram heatmap
            kde_grid_size: (x_points, y_points) for KDE grid resolution
            min_samples_for_kde: Minimum number of samples required for KDE
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.histogram_bins = histogram_bins
        self.kde_grid_size = kde_grid_size
        self.min_samples_for_kde = min_samples_for_kde

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

    def generate_player_heatmap(
        self,
        result: TrackingResult,
        player_id: int,
        method: Literal["histogram", "kde", "both"] = "both",
    ) -> dict[str, Any]:
        """
        Generate heatmap for a single player.

        Args:
            result: TrackingResult from pipeline
            player_id: Player tracker ID
            method: "histogram", "kde", or "both"

        Returns:
            Dictionary with heatmap data in format ready for JSON export
        """
        trajectory = result.get_player_trajectory(player_id)

        if len(trajectory) == 0:
            raise ValueError(f"No trajectory data found for player {player_id}")

        positions = np.array(trajectory)
        positions_meters = self._scale_to_meters(positions)

        x, y = positions_meters[:, 0], positions_meters[:, 1]

        output: dict[str, Any] = {
            "player_id": player_id,
            "sample_count": len(trajectory),
        }

        # Always try histogram (works with any amount of data)
        if method in ["histogram", "both"]:
            try:
                histogram_result = self._compute_histogram(x, y)
                output["histogram"] = histogram_result
            except Exception as e:
                print(f"Warning: Histogram failed for player {player_id}: {e}")
                # Don't include histogram key at all if it fails

        # Only attempt KDE if we have enough quality data
        if method in ["kde", "both"]:
            if len(trajectory) < self.min_samples_for_kde:
                print(
                    f"Info: Player {player_id} has only {len(trajectory)} samples "
                    f"(minimum {self.min_samples_for_kde} required for KDE). "
                    f"Skipping KDE, histogram only."
                )
                # Don't include kde key at all
            elif not self._has_sufficient_variation(x, y):
                print(
                    f"Info: Player {player_id} has insufficient spatial variation "
                    f"for KDE. Skipping KDE, histogram only."
                )
                # Don't include kde key at all
            else:
                try:
                    kde_result = self._compute_kde(x, y)
                    output["kde"] = kde_result
                except Exception as e:
                    print(f"Warning: KDE failed for player {player_id}: {e}")
                    # Don't include kde key at all if it fails

        return output

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

    def generate_team_heatmap(
        self,
        result: TrackingResult,
        team: int,
        method: Literal["histogram", "kde", "both"] = "both",
    ) -> dict[str, Any]:
        """
        Generate aggregated heatmap for entire team.

        Args:
            result: TrackingResult from pipeline
            team: Team ID (0 or 1)
            method: "histogram", "kde", or "both"
        """
        player_ids = result.get_team_players(team)

        if len(player_ids) == 0:
            raise ValueError(f"No players found for team {team}")

        # Collect all positions from all players
        all_positions = []
        for pid in player_ids:
            trajectory = result.get_player_trajectory(pid)
            all_positions.extend(trajectory)

        if len(all_positions) == 0:
            raise ValueError(f"No position data found for team {team}")

        positions = np.array(all_positions)
        positions_meters = self._scale_to_meters(positions)
        x, y = positions_meters[:, 0], positions_meters[:, 1]

        output: dict[str, Any] = {
            "team_id": team,
            "player_count": len(player_ids),
            "sample_count": len(all_positions),
        }

        # Histogram (always attempt)
        if method in ["histogram", "both"]:
            try:
                histogram_result = self._compute_histogram(x, y)
                output["histogram"] = histogram_result
            except Exception as e:
                print(f"Warning: Team histogram failed for team {team}: {e}")
                # Don't include histogram key at all if it fails

        # KDE (with quality checks)
        if method in ["kde", "both"]:
            if len(all_positions) < self.min_samples_for_kde:
                print(
                    f"Info: Team {team} has only {len(all_positions)} samples. "
                    f"Skipping KDE."
                )
                # Don't include kde key at all
            elif not self._has_sufficient_variation(x, y):
                print(f"Info: Team {team} has insufficient variation. Skipping KDE.")
                # Don't include kde key at all
            else:
                try:
                    kde_result = self._compute_kde(x, y)
                    output["kde"] = kde_result
                except Exception as e:
                    print(f"Warning: KDE failed for team {team}: {e}")
                    # Don't include kde key at all if it fails

        return output

    def generate_all_players_heatmaps(
        self,
        result: TrackingResult,
        method: Literal["histogram", "kde", "both"] = "both",
    ) -> dict[int, dict[str, Any]]:
        """
        Generate heatmaps for all players.

        Returns:
            Dictionary mapping player_id -> heatmap data
        """
        all_heatmaps = {}

        for player_id in result.get_all_player_ids():
            try:
                all_heatmaps[player_id] = self.generate_player_heatmap(
                    result, player_id, method
                )
            except ValueError as e:
                print(f"Warning: Skipping player {player_id}: {e}")

        return all_heatmaps

    def save_heatmap(
        self,
        heatmap_data: dict[str, Any],
        output_path: str,
        pretty: bool = False,
    ):
        """Save heatmap data to JSON file."""
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(heatmap_data, f, indent=2 if pretty else None)
