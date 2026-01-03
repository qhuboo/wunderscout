import csv
from pathlib import Path


class DataExporter:
    @staticmethod
    def save_csvs(tracking_data, team_assignments, total_frames, fps, output_path):
        """
        tracking_data: {frame_idx: {"ball": (x,y), "players": {id: (x,y)}}}
        team_assignments: {tracker_id: team_id}
        """
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        base_name = str(path_obj.with_suffix(""))
        home_ids = [tid for tid, team in team_assignments.items() if team == 0]
        away_ids = [tid for tid, team in team_assignments.items() if team == 1]

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

                for f_idx in range(total_frames):
                    data = tracking_data.get(f_idx, {"ball": None, "players": {}})
                    row = [1, f_idx, f"{f_idx / fps:.2f}"]
                    for tid in ids:
                        coords = data["players"].get(tid, ("NaN", "NaN"))
                        row.extend(coords)
                    row.extend(data["ball"] if data["ball"] else ("NaN", "NaN"))
                    writer.writerow(row)

        write_file(f"{base_name}_Home.csv", "Home", sorted(home_ids))
        write_file(f"{base_name}_Away.csv", "Away", sorted(away_ids))
