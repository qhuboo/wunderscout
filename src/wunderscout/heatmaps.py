import numpy as np
import json
from scipy.stats import gaussian_kde
from pathlib import Path
from typing import Optional, Literal
from .data import TrackingResult


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

    def _scale_to_meters(self, positions: np.ndarray) -> np.ndarray:
        """Convert normalized [0, 1] coordinates to meters."""
        scaled = positions.copy()
        scaled[:, 0] *= self.pitch_length
        scaled[:, 1] *= self.pitch_width
        return scaled

    def generate_player_heatmap(
        self,
        result: TrackingResult,
        player_id: int,
        method: Literal["histogram", "kde", "both"] = "both",
    ) -> dict:
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

        output = {}

        if method in ["histogram", "both"]:
            output["histogram"] = self._compute_histogram(x, y)

        if method in ["kde", "both"]:
            output["kde"] = self._compute_kde(x, y)

        return output

    def _compute_histogram(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Compute 2D histrogram heatmap."""

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

    def _compute_kde(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute KDE smoothed density field.

        Returns dict with:
            - x: 1D list of x coordinates
            - y: 1D list of y coordinates
            - values: 2D list where values[i][j] = density at [x[j], y[i]]
        """
        values = np.vstack([x, y])
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
            "values": Z.tolist(),  # Shape: (len(y), len(x))
        }

    def generate_team_heatmap(
        self,
        result: TrackingResult,
        team: int,
        method: Literal["histogram", "kde", "both"] = "both",
    ) -> dict:
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

        positions = np.array(all_positions)
        positions_meters = self._scale_to_meters(positions)
        x, y = positions_meters[:, 0], positions_meters[:, 1]

        output = {}

        if method in ["histogram", "both"]:
            output["histogram"] = self._compute_histogram(x, y)

        if method in ["kde", "both"]:
            output["kde"] = self._compute_kde(x, y)

        return output

    def generate_all_players_heatmaps(
        self,
        result: TrackingResult,
        method: Literal["histogram", "kde", "both"] = "both",
    ) -> dict[int, dict]:
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
        heatmap_data: dict,
        output_path: str,
        pretty: bool = False,
    ):
        """Save heatmap data to JSON file."""
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(heatmap_data, f, indent=2 if pretty else None)
