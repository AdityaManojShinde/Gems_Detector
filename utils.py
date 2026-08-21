import threading
import time

import cv2

SMOOTHING_ALPHA = 0.4
HOLD_FRAMES = 8
OVERLAP_IOU_THRESH = 0.45

# Pipeline building blocks


class FreshestFrame:
    """
    Reads frames from a VideoCapture in a background thread and always
    exposes only the MOST RECENT frame, dropping stale ones so delay
    never accumulates.
    """

    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            if not self.cap.isOpened():
                print("Stream not open, reconnecting...")
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("Frame read failed, reconnecting...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            frame = self.frame
            self.frame = None
        return frame

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()

    @property
    def is_finished(self):
        # A live camera/IP stream doesn't have a natural "end" - the
        # worker just keeps waiting for the next frame.
        return False


class FileFrame:
    """
    Reads frames sequentially from an uploaded video file. Unlike
    FreshestFrame this does NOT drop frames (there's no "freshest" frame
    to chase for a file that isn't live) - every frame gets processed
    once, in order. Sets `finished` once the file is exhausted so the
    caller can stop the pipeline instead of polling forever.
    """

    def __init__(self, path, loop=False):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        self.finished = False
        self.stopped = False

    def read(self):
        if self.stopped:
            return None

        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    return frame
            self.finished = True
            return None

        return frame

    def stop(self):
        self.stopped = True
        self.cap.release()

    @property
    def is_finished(self):
        return self.finished


class BoxSmoother:
    """
    Keeps a smoothed bounding box per tracker ID and holds it on screen
    for a short grace period after the tracker briefly loses the object.

    Also de-duplicates tracks whose boxes overlap heavily (see
    `_dedupe_overlapping_tracks`) - this is the fix for boxes that
    "double up" for a beat whenever ByteTrack swaps a track's ID.
    """

    def __init__(
        self,
        alpha=SMOOTHING_ALPHA,
        hold_frames=HOLD_FRAMES,
        overlap_iou_thresh=OVERLAP_IOU_THRESH,
    ):
        self.alpha = alpha
        self.hold_frames = hold_frames
        self.overlap_iou_thresh = overlap_iou_thresh
        self.tracks = {}  # track_id -> {"box", "cls_name", "conf", "age"}

    def update(self, detections):
        seen_ids = set()

        for track_id, cls_name, conf, box in detections:
            if track_id is None:
                continue

            seen_ids.add(track_id)
            if track_id in self.tracks:
                old_box = self.tracks[track_id]["box"]
                box = tuple(
                    self.alpha * new + (1 - self.alpha) * old
                    for new, old in zip(box, old_box)
                )

            self.tracks[track_id] = {
                "box": box,
                "cls_name": cls_name,
                "conf": conf,
                "age": 0,
            }

        stale_ids = []
        for track_id, data in self.tracks.items():
            if track_id not in seen_ids:
                data["age"] += 1
                if data["age"] > self.hold_frames:
                    stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.tracks[track_id]

        self._dedupe_overlapping_tracks()

        held = [
            (track_id, data["cls_name"], data["conf"], data["box"])
            for track_id, data in self.tracks.items()
        ]
        untracked = [d for d in detections if d[0] is None]
        return held + untracked

    def _dedupe_overlapping_tracks(self):
        """
        When ByteTrack drops a track and assigns a new ID to the same
        physical object a frame or two later (common with brief
        occlusion or motion blur), the old held box and the new box can
        both be on screen for up to `hold_frames` frames. Collapse any
        pair of tracks whose boxes overlap above `overlap_iou_thresh`,
        keeping the fresher (lower age) / higher-confidence one.
        """
        ids = list(self.tracks.keys())
        to_delete = set()
        for i in range(len(ids)):
            id_a = ids[i]
            if id_a in to_delete:
                continue
            for j in range(i + 1, len(ids)):
                id_b = ids[j]
                if id_b in to_delete:
                    continue
                a, b = self.tracks[id_a], self.tracks[id_b]
                if iou(a["box"], b["box"]) < self.overlap_iou_thresh:
                    continue
                a_key = (a["age"], -a["conf"])
                b_key = (b["age"], -b["conf"])
                to_delete.add(id_b if a_key <= b_key else id_a)
        for tid in to_delete:
            del self.tracks[tid]


# Shared state + background inference worker


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg_bytes = None
        self.frame_seq = 0
        self.total_count = 0
        self.tablet_count = 0
        self.broken_count = 0
        self.running = False
        self.error = None
        self.mode = "live"  # "live" | "video" | "image"


def iou(box_a, box_b):
    """Standard IOU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def extract_detections(result, conf_threshold, class_conf_thresholds):
    """Filter result boxes by confidence and return detection tuples."""
    names = result.names
    detections = []

    for box in result.boxes:
        conf = float(box.conf)
        cls_id = int(box.cls)
        cls_name = names[cls_id]

        threshold = class_conf_thresholds.get(cls_name, conf_threshold)
        if conf < threshold:
            continue

        track_id = int(box.id) if box.id is not None else None
        xyxy = tuple(map(float, box.xyxy[0].tolist()))
        detections.append((track_id, cls_name, conf, xyxy))

    return detections


def resize_for_display(frame, max_width):
    """Downscale a frame before JPEG encoding it for the browser."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA
    )


def draw_boxes(frame, detections, class_colors, default_color):
    """Draw boxes and labels in place and return the frame with class counts."""
    counts = {}

    for track_id, cls_name, conf, box in detections:
        counts[cls_name] = counts.get(cls_name, 0) + 1
        x1, y1, x2, y2 = map(int, box)

        color = class_colors.get(cls_name, default_color)
        label = f"{cls_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 8),
            (x1 + text_width + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return frame, counts
