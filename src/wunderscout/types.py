from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import csv
import numpy as np
import supervision as sv

import logging

logger = logging.getLogger(__name__)


class ClassId(Enum):
    BALL = 0
    GOALKEEPER = 1
    PLAYER = 2
    REFEREE = 3


@dataclass
class SaveResult:
    successful_paths: list[Path] = field(default_factory=list)
    failed_paths: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class Frames:
    _detections: list[sv.Detections]
    _player_team_map: dict[int, int] = field(init=False)

    def __post_init__(self):
        self._player_team_map = self._build_player_team_map()

    def __len__(self):
        return len(self._detections)

    def __getitem__(self, key):
        return self._detections[key]

    def __iter__(self):
        return iter(self._detections)

    def _build_player_team_map(self):
        merged = sv.Detections.merge(self._detections)
        players = merged[merged.class_id == ClassId.PLAYER.value]
        player_team_map = {
            int(tracker_id): int(team_id)
            for tracker_id, team_id in zip(players.tracker_id, players.data["team_id"])
        }

        return player_team_map

    def get_all_player_ids(self):
        return list(self._player_team_map.keys())

    def get_all_team_ids(self):
        return list(set(self._player_team_map.values()))

    def get_team_for_player(self, player_id: int):
        return self._player_team_map[player_id]

    def save_csvs(self, output_path: str | Path) -> SaveResult:
        """Export tracking data to CSV files (one per team)."""
        path_obj = Path(output_path)
        path_obj.mkdir(parents=True, exist_ok=True)

        team_ids = list(self.get_all_team_ids())
        team_players = {
            team_id: [
                pid for pid, tid in self._player_team_map.items() if tid == team_id
            ]
            for team_id in team_ids
        }

        def write_file(path, team_name, player_ids):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)

                # Row 1: Team names
                writer.writerow(
                    ["", "", ""]
                    + [item for _ in player_ids for item in (team_name, "")]
                    + ["", ""]
                )
                writer.writerow(
                    ["", "", ""]
                    + [item for pid in player_ids for item in (str(pid), "")]
                    + ["", ""]
                )
                writer.writerow(
                    ["Period", "Frame", "Time [s]"]
                    + [item for pid in player_ids for item in (f"Player{pid}", "")]
                    + ["Ball", ""]
                )

                for f_idx, detection in enumerate(self._detections):
                    # period, frame, time
                    row = [1, f_idx, ""]

                    # Get player positions
                    for player_id in player_ids:
                        if detection.tracker_id is not None:
                            mask = detection.tracker_id == player_id
                            if mask.any():
                                idx = np.where(mask)[0][0]
                                coords = detection.data["pitch_coordinates"][idx]
                                row.extend([f"{coords[0]:.5f}", f"{coords[1]:.5f}"])
                            else:
                                row.extend(["NaN", "NaN"])
                        else:
                            row.extend(["NaN", "NaN"])

                    # Get ball position
                    ball_mask = detection.class_id == ClassId.BALL
                    if ball_mask.any():
                        idx = np.where(ball_mask)[0][0]
                        coords = detection.data["pitch_coordinates"][idx]
                        row.extend([f"{coords[0]:.5f}", f"{coords[1]:.5f}"])
                    else:
                        row.extend(["NaN", "NaN"])

                    writer.writerow(row)

        save_result = SaveResult()

        for team_id in team_ids:
            path = path_obj / f"team_{team_id}.csv"
            try:
                write_file(path, f"Team{team_id}", team_players[team_id])
                save_result.successful_paths.append(path)
            except Exception as e:
                error_msg = f"Unexpected error saving Team{team_id}.csv: {e}"
                save_result.errors.append(str(error_msg))
                save_result.failed_paths.append(path)

        return save_result
