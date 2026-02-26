import csv
from pathlib import Path
from .data import TrackingResult


class DataExporter:
    @staticmethod
    def save_csvs(result: TrackingResult, output_path: str) -> list[str]:
        """Export tracking data to CSV files (one per team)."""
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)

        team1_ids = result.get_team_players(0)
        team2_ids = result.get_team_players(1)

        def write_file(filename, team_name, ids):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["", "", ""] + [team_name for _ in ids for _ in (0, 1)] + ["", ""]
                )
                writer.writerow(
                    ["", "", ""] + [str(pid) for pid in ids for _ in (0, 1)] + ["", ""]
                )
                writer.writerow(
                    ["Period", "Frame", "Time [s]"]
                    + [f"Player{pid}_{axis}" for pid in ids for axis in ("X", "Y")]
                    + ["Ball_X", "Ball_Y"]
                )

                for f_idx in range(result.total_frames):
                    data = result.frames.get(f_idx, {"ball": None, "players": {}})
                    row = [1, f_idx, f"{f_idx / result.fps:.2f}"]
                    for tid in ids:
                        coords = data["players"].get(tid, ("NaN", "NaN"))
                        row.extend(coords)
                    row.extend(data["ball"] if data["ball"] else ("NaN", "NaN"))
                    writer.writerow(row)

        created = []

        write_file(out / "Team1.csv", "Team1", sorted(team1_ids))
        created.append(str(out / "Team1.csv"))

        write_file(out / "Team2.csv", "Team2", sorted(team2_ids))
        created.append(str(out / "Team2.csv"))

        return created
