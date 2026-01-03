import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from pathlib import Path

# Load match events JSON (replace with your actual file path)
with open(
    "./data/3825818.json",
    "r",
) as f:
    events = json.load(f)

# Build mapping from player ID -> player name from the Starting XI event
player_id_to_name = {}

for ev in events:
    if ev["type"]["name"] == "Starting XI":
        for lineup in ev["tactics"]["lineup"]:
            pid = lineup["player"]["id"]
            name = lineup["player"]["name"]
            player_id_to_name[pid] = name

# Data structures for passes and positions
edges = defaultdict(int)  # (passer, recipient) -> count of passes
player_positions = defaultdict(list)  # player_id -> list of [x, y] positions

TEAM_NAME = "Real Sociedad"

# Extract completed passes
for ev in events:
    if ev["type"]["name"] == "Pass" and ev["team"]["name"] == TEAM_NAME:
        passer = ev["player"]["id"]
        recipient = ev.get("pass", {}).get("recipient", {}).get("id")
        outcome = ev.get("pass", {}).get("outcome", {"name": "Complete"})["name"]

        if outcome == "Complete" and recipient is not None:
            edges[(passer, recipient)] += 1
            start = ev["location"]
            end = ev["pass"]["end_location"]
            player_positions[passer].append(start)
            player_positions[recipient].append(end)

# Calculate avg positions
avg_positions = {}
for player_id, coords in player_positions.items():
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    avg_positions[player_id] = [sum(xs) / len(xs), sum(ys) / len(ys)]

# Build a JSON-friendly structure for export (nodes + links)
nodes = [{"id": pid, "x": pos[0], "y": pos[1]} for pid, pos in avg_positions.items()]
links = [
    {"source": src, "target": tgt, "value": count}
    for (src, tgt), count in edges.items()
]

network = {"nodes": nodes, "links": links}
os.makedirs("pass_network", exist_ok=True)
with open("./pass_network/pass_network.json", "w") as f:
    json.dump(network, f, indent=2)

# Build NetworkX graph
G = nx.DiGraph()

# Add nodes with positions
for pid, pos in avg_positions.items():
    G.add_node(pid, pos=(pos[0], pos[1]))

# Add edges with weights
for (src, tgt), count in edges.items():
    G.add_edge(src, tgt, weight=count)

# Draw graph
pos = nx.get_node_attributes(G, "pos")
labels = {pid: player_id_to_name.get(pid, str(pid)) for pid in G.nodes()}

fig, ax = plt.subplots(figsize=(10, 7))

# Draw pitch outline
ax.set_xlim(0, 120)
ax.set_ylim(0, 80)
ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], color="black")

# Draw nodes
nx.draw_networkx_nodes(G, pos, ax=ax, node_color="skyblue", node_size=500)

# Draw edges
nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    width=[d["weight"] * 0.2 for _, _, d in G.edges(data=True)],
    alpha=0.7,
    arrowsize=10,
)

# Draw player names
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

plt.title("Team Pass Network")
plt.savefig("./pass_network/pass_network_viz.png", dpi=150, bbox_inches="tight")
