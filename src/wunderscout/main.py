import os
from dotenv import load_dotenv
from roboflow import Roboflow
import torch
from ultralytics import YOLO
import supervision as sv
import cv2
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import numpy as np
import umap
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import base64
from io import BytesIO

load_dotenv()

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"


def train():
    print("Beginning training ...")
    # This is to get the roboflow dataset and train.
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(20)
    dataset = version.download("yolov11")

    model_base_s = YOLO("yolo11s.pt")

    results = model_base_s.train(
        data="./football-players-detection-20/data.yaml",
        save=True,
        epochs=50,
        imgsz=1280,
        plots=True,
        device=0,
        batch=6,
        project="runs/detect",
    )


def inference(video_path):
    print("Inference 3 ...")
    # We do another inference with different annotations
    BALL_ID = 0

    model_trained = YOLO("./runs/detect/train/weights/best.pt")
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "FFD700"]), thickness=2
    )

    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"), base=25, height=21, outline_thickness=1
    )

    frame_generator_2 = sv.get_video_frames_generator(video_path)
    frame_2 = next(frame_generator_2)

    result_3 = model_trained.predict(
        frame_2, save=True, project="runs", name="inference"
    )[0]
    detections_2 = sv.Detections.from_ultralytics(result_3)

    ball_detections = detections_2[detections_2.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections_2[detections_2.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections.class_id -= 1

    annotated_frame_2 = frame_2.copy()
    annotated_frame_2 = ellipse_annotator.annotate(
        scene=annotated_frame_2, detections=all_detections
    )
    annotated_frame_2 = triangle_annotator.annotate(
        scene=annotated_frame_2, detections=ball_detections
    )

    cv2.imwrite("./runs/annotated_frame.jpg", annotated_frame_2)


def inference_with_player_tracking(video_path):
    print("Inference with tracking ...")
    # We will do inference again but this time with the video.
    BALL_ID = 0

    model_trained = YOLO("./runs/detect/train/weights/best.pt")
    ellipse_annotator_2 = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]), thickness=2
    )

    label_annotator_2 = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        text_color=sv.Color.from_hex("#000000"),
        text_position=sv.Position.BOTTOM_CENTER,
    )

    triangle_annotator_2 = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"),
        base=25,
        height=21,
        outline_thickness=1,
    )

    tracker = sv.ByteTrack()

    frame_generator_3 = sv.get_video_frames_generator(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "./runs/video_tracked.mp4", fourcc, fps, (frame_width, frame_height)
    )

    for frame in frame_generator_3:
        result_tracking = model_trained.predict(frame, conf=0.3)[0]

        detections = sv.Detections.from_ultralytics(result_tracking)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections.class_id -= 1
        all_detections = tracker.update_with_detections(detections=all_detections)

        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator_2.annotate(
            scene=annotated_frame, detections=all_detections
        )
        annotated_frame = label_annotator_2.annotate(
            scene=annotated_frame, detections=all_detections, labels=labels
        )
        annotated_frame = triangle_annotator_2.annotate(
            scene=annotated_frame, detections=ball_detections
        )

        out.write(annotated_frame)
    out.release()


def create_player_crops(video_path):
    print("Generating player crops ...")
    # Getting training data for cluster model
    PLAYER_ID = 2
    STRIDE = 30

    model_trained = YOLO("./runs/detect/train/weights/best.pt")
    frame_generator_4 = sv.get_video_frames_generator(
        source_path=video_path, stride=STRIDE
    )

    crops = []
    for frame in tqdm(frame_generator_4, desc="collecting crops"):
        result = model_trained.predict(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    print(f"Total crops collected: {len(crops)}.")

    # Save to a file
    # out_dir = "./runs/player_crops"
    # os.makedirs(out_dir, exist_ok=True)

    # for i, crop in enumerate(crops):
    #     filename = os.path.join(out_dir, f"crop_{i:04d}.jpg")
    #     cv2.imwrite(filename, crop)

    # Convert to PIL
    pil_crops = [sv.cv2_to_pillow(c) for c in crops]
    return pil_crops


def extract_embeddings(crops):
    BATCH_SIZE = 32

    batches = chunked(crops, BATCH_SIZE)
    data = []

    with torch.no_grad():
        for batch in tqdm(batches, desc="embedding extraction"):
            inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
            outputs = EMBEDDINGS_MODEL(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            data.append(embeddings)
    data = np.concatenate(data)
    return data


def cluster_players_by_team(embeddings):
    REDUCER = umap.UMAP(n_components=3)
    CLUSTERING_MODEL = KMeans(n_clusters=2, n_init=10, random_state=42)

    projections = REDUCER.fit_transform(embeddings)
    clusters = CLUSTERING_MODEL.fit_predict(projections)

    return projections, clusters


def save_projection_plot_html(projections, clusters, crops):
    # inline helper: convert a crop (PIL image) to base64 string
    def pil_image_to_data_uri(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    # convert crops to base64-encoded URIs
    image_data_uris = {
        f"image_{i}": pil_image_to_data_uri(crop) for i, crop in enumerate(crops)
    }
    image_ids = np.array([f"image_{i}" for i in range(len(crops))])

    traces = []
    unique_clusters = np.unique(clusters)
    for unique_cluster in unique_clusters:
        mask = clusters == unique_cluster
        trace = go.Scatter3d(
            x=projections[mask][:, 0],
            y=projections[mask][:, 1],
            z=projections[mask][:, 2],
            mode="markers+text",  # hard-coded: markers+text
            text=clusters[mask],
            customdata=image_ids[mask],
            name=str(unique_cluster),
            marker=dict(size=6),
            hovertemplate="<b>class: %{text}</b><br>image ID: %{customdata}<extra></extra>",
        )
        traces.append(trace)

    # make cube axis range
    min_val = np.min(projections)
    max_val = np.max(projections)
    padding = (max_val - min_val) * 0.05
    axis_range = [min_val - padding, max_val + padding]

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=axis_range),
            yaxis=dict(title="Y", range=axis_range),
            zaxis=dict(title="Z", range=axis_range),
            aspectmode="cube",
        ),
        width=1000,
        height=1000,
        showlegend=True,
    )

    # embed chart HTML with custom JS for crop preview
    plotly_div = fig.to_html(
        full_html=False, include_plotlyjs=True, div_id="scatter-plot-3d"
    )
    javascript_code = f"""
    <script>
        function displayImage(imageId) {{
            var imageElement = document.getElementById('image-display');
            var placeholderText = document.getElementById('placeholder-text');
            var imageDataURIs = {image_data_uris};
            imageElement.src = imageDataURIs[imageId];
            imageElement.style.display = 'block';
            placeholderText.style.display = 'none';
        }}

        var chartElement = document.getElementById('scatter-plot-3d');
        chartElement.on('plotly_click', function(data) {{
            var customdata = data.points[0].customdata;
            displayImage(customdata);
        }});
    </script>
    """

    html_template = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                #image-container {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 200px;
                    height: 200px;
                    padding: 5px;
                    border: 1px solid #ccc;
                    background-color: white;
                    z-index: 1000;
                    box-sizing: border-box;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                }}
                #image-display {{
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }}
            </style>
        </head>
        <body>
            {plotly_div}
            <div id="image-container">
                <img id="image-display" src="" alt="Selected image" style="display: none;" />
                <p id="placeholder-text">Click a data point to display the image</p>
            </div>
            {javascript_code}
        </body>
    </html>
    """

    out_path = "./runs/player_clusters.html"
    with open(out_path, "w") as f:
        f.write(html_template)

    print(f"Interactive plot saved to {out_path}. Open it in your browser.")


def resolve_goalkeepers_team_id(players, goalkeepers):
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []

    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)


def inference_with_goalkeepers(video_path):
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    crops = create_player_crops(video_path)
    embeddings = extract_embeddings(crops)

    REDUCER = umap.UMAP(n_components=3)
    CLUSTERING_MODEL = KMeans(n_clusters=2, n_init=10, random_state=42)

    projections = REDUCER.fit_transform(embeddings)
    clustering_model = CLUSTERING_MODEL.fit(projections)

    model_trained = YOLO("./runs/detect/train/weights/best.pt")

    frame_generator = sv.get_video_frames_generator(video_path)
    frame = next(frame_generator)

    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "./runs/video_tracked_2.mp4", fourcc, fps, (frame_width, frame_height)
    )

    for frame in frame_generator:
        result = model_trained.predict(frame, conf=0.3)[0]

        detections = sv.Detections.from_ultralytics(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        goalkeepers_detections = all_detections[
            all_detections.class_id == GOALKEEPER_ID
        ]
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        player_embeddings = extract_embeddings(players_crops)
        player_projection = REDUCER.transform(player_embeddings)
        players_detections.class_id = clustering_model.predict(player_projection)

        goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
            players_detections, goalkeepers_detections
        )

        referees_detections.class_id -= 1

        all_detections = sv.Detections.merge(
            [players_detections, goalkeepers_detections, referees_detections]
        )

        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

        all_detections.class_id = all_detections.class_id.astype(int)

        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
            thickness=2,
        )
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
            text_color=sv.Color.from_hex("#000000"),
            text_position=sv.Position.BOTTOM_CENTER,
        )
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"), base=25, height=21, outline_thickness=1
        )

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame, detections=all_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=all_detections, labels=labels
        )
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame, detections=ball_detections
        )

        out.write(annotated_frame)
    out.release()


def keypoint_detection_train():
    print("Beginning training ...")
    # This is to get the roboflow dataset and train.
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
    version = project.version(15)
    dataset = version.download("yolov8")

    model_base_s = YOLO("yolo11m-pose.pt")
    results = model_base_s.train(
        data="./football-field-detection-15/data.yaml",
        save=True,
        epochs=300,
        plots=True,
        imgsz=1080,
        device=0,
        batch=2,
        project="runs/keypoint_detect/",
    )


def keypoint_detection_inference(video_path: str) -> None:
    model = YOLO("./runs/keypoint_detect/train4/weights/best.pt")
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex("#FF1493"),
        radius=8,
    )

    frame = next(sv.get_video_frames_generator(video_path))
    res = model.predict(frame, conf=0.3, iou=0.7, imgsz=1088, verbose=False)[0]

    if res.boxes is None or len(res.boxes) == 0:
        cv2.imwrite("./runs/annotated_frame_keypoint.jpg", frame)
        return

    best_i = int(np.argmax(res.boxes.conf.cpu().numpy()))

    kxy = res.keypoints.xy[best_i].cpu().numpy()
    kconf = res.keypoints.conf[best_i].cpu().numpy()

    keep = kconf > 0.5
    kxy = kxy[keep]

    kp = sv.KeyPoints(xy=kxy[np.newaxis, ...])

    out = vertex_annotator.annotate(scene=frame.copy(), key_points=kp)
    cv2.imwrite("./runs/annotated_frame_keypoint.jpg", out)


def keypoint_detection_indices(video_path, output_path="keypoint_debug.mp4"):
    print(f"DEBUG: Starting keypoint visualization for {video_path}")

    model_trained = YOLO("./runs/keypoint_detect/train4/weights/best.pt")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...")

        result = model_trained.predict(frame, conf=0.5, verbose=False)[0]

        if result.keypoints is not None and result.keypoints.xy.shape[0] > 0:
            kpts_xy = result.keypoints.xy[0].cpu().numpy()
            kpts_conf = result.keypoints.conf[0].cpu().numpy()

            for i, (x, y) in enumerate(kpts_xy):
                conf = kpts_conf[i]

                if conf < 0.5 or (x == 0 and y == 0):
                    continue

                x, y = int(x), int(y)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(i + 1),
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
        else:
            cv2.putText(
                frame,
                "NO PITCH DETECTED",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        out.write(frame)

    cap.release()
    out.release()
    print("Video generation complete.")


def keypoint_detection_val_test(model_path, data_yaml):
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split="test", plots=True)

    print(results)


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    video_path = "/home/lucas/Documents/dev/local/yolo_torch/video.mp4"

    # train()
    # inference(video_path)
    # inference_with_player_tracking(video_path)
    # crops = create_player_crops(video_path)
    # embeddings = extract_embeddings(crops)
    # projections, clusters = cluster_players_by_team(embeddings)
    # save_projection_plot_html(projections, clusters, crops)
    # inference_with_goalkeepers(video_path)
    # keypoint_detection_train()
    # keypoint_detection_inference(video_path)
    keypoint_detection_indices(video_path)
    # val_test(
    #     "./runs/keypoint_detect/train4/weights/best.pt",
    #     "./football-field-detection-15/data.yaml",
    # )


if __name__ == "__main__":
    main()
