from dataclasses import dataclass


@dataclass
class TrackingResult:
    frames: dict[int, dict]
    team_assignments: dict[int, int]
    total_frames: int
    fps: float

    def get_team_players(self, team: int) -> list[int]:
        """Get player IDs for a specific team (0 or 1)."""
        return [tid for tid, t in self.team_assignments.items() if t == team]

    def get_all_player_ids(self) -> list[int]:
        """Get all player IDs."""
        return list(self.team_assignments.keys())

    def get_player_trajectory(self, player_id: int) -> list[tuple[float, float]]:
        """Get all positions for one player."""
        return [
            self.frames[f]["players"][player_id]
            for f in sorted(self.frames.keys())
            if player_id in self.frames[f]["players"]
        ]

    def get_ball_trajectory(self) -> list[tuple[float, float]]:
        """Get all ball positions."""
        return [
            self.frames[f]["ball"]
            for f in sorted(self.frames.keys())
            if self.frames[f]["ball"] is not None
        ]
