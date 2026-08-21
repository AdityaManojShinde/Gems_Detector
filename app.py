"""
Tablet quality-control YOLO pipeline 
- custom frontend served from a
gradio.Server / FastAPI backend (no default gr.Blocks UI).

Run with:
    python app.py
    (or: uvicorn app:app --host 0.0.0.0 --port 7860)
"""

import os
import threading
import time

import anyio
import cv2
import torch
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO

from utils import (
    BoxSmoother,
    FreshestFrame,
    SharedState,
    draw_boxes,
    extract_detections,
    resize_for_display,
)

try:
 
    from gradio import Server

    app = Server()
except ImportError:
    from fastapi import FastAPI

    app = FastAPI()
    print(
        "gradio.Server not found in this gradio version - continuing with "
        "plain FastAPI (identical behavior for this app). "
        "`pip install -U gradio` if you specifically want gradio.Server."
    )

# ============================================================
# CONFIG - edit these values directly
# ============================================================
MODEL_PATH = "./model/best.pt"       # path to trained YOLO .pt file
STREAM_URL = "http://192.168.1.4:8080/video"                # IP webcam stream URL
CONF_THRESHOLD = 0.4                                        # confidence threshold
IMGSZ = 640                                                 # inference image size
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = True                                       # fp16 inference (GPU only; auto-disabled on CPU below)
TRACKER_CONFIG = "bytetrack.yaml"                           # built-in ultralytics tracker
TABLET_CLASS_NAME = "tablet"                                 # class name counted as "Tablet Count"
BROKEN_CLASS_NAME = "damaged_tablet"                         # class name counted as "Broken Count"


AGNOSTIC_NMS = True
NMS_IOU_THRESHOLD = 0.5   
                      

DISPLAY_MAX_WIDTH = 640                          # downscale frames before JPEG-encoding
TARGET_DISPLAY_FPS = 20                          # cap the encode rate so JPEG-encoding
                                  

USE_HALF = HALF_PRECISION and DEVICE.startswith("cuda")

QUANTIZE = 16 if USE_HALF else None

# Color per class (BGR) 
CLASS_COLORS = {
    "tablet": (0, 200, 0),          # green
    "damaged_tablet": (0, 0, 255),  # red
}
DEFAULT_COLOR = (255, 255, 0)

# Per-class confidence thresholds 
CLASS_CONF_THRESHOLDS = {
    "tablet": 0.4,
    "damaged_tablet": 0.6,
}

PREDICT_CONF = min([CONF_THRESHOLD, *CLASS_CONF_THRESHOLDS.values()])





shared = SharedState()
_model_holder = {"model": None}
_worker_thread = None
_stop_event = threading.Event()


def _load_model():
    if _model_holder["model"] is None:
        print(f"Loading model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        _model_holder["model"] = model
        print(f"Running inference on: {DEVICE} (quantize: {QUANTIZE or 'fp32'})")
    return _model_holder["model"]


def _inference_worker(stop_event):
    """
    Owns the camera connection and the YOLO/tracker/smoother pipeline.
    Runs independently of any HTTP request - starting/stopping/refreshing
    browser tabs never touches this loop.
    """
    try:
        model = _load_model()
    except FileNotFoundError as exc:  # bad MODEL_PATH, etc.
        with shared.lock:
            shared.error = f"Failed to load model: {exc}"
            shared.running = False
        return

    print(f"Connecting to stream: {STREAM_URL}")
    stream = FreshestFrame(STREAM_URL)
    smoother = BoxSmoother()

    min_interval = 1.0 / TARGET_DISPLAY_FPS
    last_encode = 0.0

    try:
        while not stop_event.is_set():
            frame = stream.read()
            if frame is None:
                time.sleep(0.005)
                continue

            results = model.track(
                frame,
                conf=PREDICT_CONF,
                imgsz=IMGSZ,
                device=DEVICE,
                quantize=QUANTIZE,
                agnostic_nms=AGNOSTIC_NMS,
                iou=NMS_IOU_THRESHOLD,
                tracker=TRACKER_CONFIG,
                persist=True,
                verbose=False,
            )
            result = results[0]

            detections = extract_detections(
                result, CONF_THRESHOLD, CLASS_CONF_THRESHOLDS
            )
            smoothed = smoother.update(detections)
            annotated, counts = draw_boxes(
                frame, smoothed, CLASS_COLORS, DEFAULT_COLOR
            )

            now = time.time()
            if now - last_encode < min_interval:
                continue
            last_encode = now

            small = resize_for_display(annotated, DISPLAY_MAX_WIDTH)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            with shared.lock:
                shared.jpeg_bytes = buf.tobytes()
                shared.frame_seq += 1
                shared.total_count = sum(counts.values())
                shared.tablet_count = counts.get(TABLET_CLASS_NAME, 0)
                shared.broken_count = counts.get(BROKEN_CLASS_NAME, 0)
                shared.error = None
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        with shared.lock:
            shared.error = f"Pipeline error: {exc}"
    finally:
        stream.stop()
        with shared.lock:
            shared.running = False
            shared.jpeg_bytes = None
        print("Inference worker stopped, camera released.")


def _mjpeg_generator():
    last_seq_sent = -1
    while True:
        with shared.lock:
            running = shared.running
            seq = shared.frame_seq
            frame_bytes = shared.jpeg_bytes
        if not running:
            break
        if frame_bytes is None or seq == last_seq_sent:
            time.sleep(0.005)
            continue
        last_seq_sent = seq
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
            + frame_bytes + b"\r\n"
        )



@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    async with await anyio.open_file(html_path, "r", encoding="utf-8") as f:
        return await f.read()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/counts")
def counts():
    with shared.lock:
        return JSONResponse({
            "total": shared.total_count,
            "tablet": shared.tablet_count,
            "broken": shared.broken_count,
            "running": shared.running,
            "error": shared.error,
        })


@app.post("/start")
def start_pipeline():
    global _worker_thread, _stop_event
    with shared.lock:
        if shared.running:
            return JSONResponse({"status": "already_running"})
        shared.running = True
        shared.error = None
    _stop_event = threading.Event()
    _worker_thread = threading.Thread(target=_inference_worker, args=(_stop_event,), daemon=True)
    _worker_thread.start()
    return JSONResponse({"status": "started"})


@app.post("/stop")
def stop_pipeline():
    _stop_event.set()
    with shared.lock:
        shared.running = False
    return JSONResponse({"status": "stopping"})


if __name__ == "__main__":
    app.launch( server_port=7860, show_error=True)