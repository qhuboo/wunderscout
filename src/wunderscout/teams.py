import numpy as np
import umap
from sklearn.cluster import KMeans
import supervision as sv


class TeamClassifier:
    def __init__(self):
        self.reducer = umap.UMAP(n_components=3)
        self.clusterer = KMeans(n_clusters=2, n_init=10, random_state=42)
        self.history = {}

    def fit(self, embeddings):
        projections = self.reducer.fit_transform(embeddings)
        self.clusterer.fit(projections)

    def get_consensus_team(self, tracker_id, embedding):
        proj = self.reducer.transform(embedding.reshape(1, -1))
        pred = self.clusterer.predict(proj)[0]

        if tracker_id not in self.history:
            self.history[tracker_id] = []
        self.history[tracker_id].append(pred)
        if len(self.history[tracker_id]) > 50:
            self.history[tracker_id].pop(0)

        return (
            1
            if (sum(self.history[tracker_id]) / len(self.history[tracker_id])) > 0.5
            else 0
        )

    def resolve_goalkeepers_team_id(self, players, goalkeepers):
        """
        Assigns goalkeepers to the team whose centroid is closest.
        players: sv.Detections (already classified with class_id 0 or 1)
        goalkeepers: sv.Detections
        """
        if len(players) == 0 or len(goalkeepers) == 0:
            return np.array([0] * len(goalkeepers))

        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        # Calculate centroids for Team 0 and Team 1
        team_0_mask = players.class_id == 0
        team_1_mask = players.class_id == 1

        # Handle cases where one team might not be detected yet
        if np.any(team_0_mask):
            team_0_centroid = players_xy[team_0_mask].mean(axis=0)
        else:
            team_0_centroid = np.array([0, 0])

        if np.any(team_1_mask):
            team_1_centroid = players_xy[team_1_mask].mean(axis=0)
        else:
            team_1_centroid = np.array([10000, 10000])  # Far away

        goalkeepers_team_id = []

        for gk_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
            dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

        return np.array(goalkeepers_team_id)

    def get_final_assignments(self):
        assignments = {}
        for tid, votes in self.history.items():
            if len(votes) > 0:
                avg = sum(votes) / len(votes)
                assignments[tid] = 1 if avg > 0.5 else 0
        return assignments

