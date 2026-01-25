from dataclasses import dataclass


@dataclass
class TrackingResult:
    frames: dict[int, dict]
    team_assignments: dict[int, int]
    total_frames: int
    fps: float

    # frames shape
    # {
    #     0: {
    #         "players": {
    #             5: (0.234, 0.567),    # tracker_id: (normalized_x, normalized_y)
    #             12: (0.789, 0.123),
    #             ...
    #         },
    #         "ball": (0.5, 0.5) or None  # (normalized_x, normalized_y) or None
    #     },
    #     1: {
    #         "players": {...},
    #         "ball": ...
    #     },
    #     ...
    # }

    # team_assignments shape
    # {
    #     5: 0,     # tracker_id 5 -> Team 0
    #     12: 1,    # tracker_id 12 -> Team 1
    #     18: 0,
    #     ...
    # }

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
