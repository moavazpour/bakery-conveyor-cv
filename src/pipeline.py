"""
Core system logic: Detection (FastSAM) + Tracking + Counting + Online
Clustering (DINOv2).

Key design points:
    1. SINGLE PASS: the video is processed exactly once. A new category
       starts unconfirmed ("new category" on screen). As soon as it
       gets a second member, it is confirmed and immediately gets a
       fixed number ("Produktart N"). Categories that never get a
       second member are merged into "Unklar / Sonstige" only in the
       final summary.
    2. Automatic detection of the belt's motion direction (a short,
       lightweight pre-pass over the first few hundred frames, before
       the real single pass starts), works for ANY angle, not just
       horizontal/vertical.
    3. A separate column is appended next to the video to show detection results,
       so nothing overlaps the product footage.
    4. GPU is independently selectable for DINOv2 (clustering) vs.
       FastSAM (detection).
"""

import os
import json
import shutil
import time
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ======================================================================
# BASIC HELPERS
# ======================================================================
def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def polygon_area(points):
    pts = np.array(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def point_in_polygon(point, polygon):
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0


def extract_masked_crop(frame, bbox, mask, padding=5):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
    x2p, y2p = min(w, x2 + padding), min(h, y2 + padding)
    crop = frame[y1p:y2p, x1p:x2p].copy()
    if mask is not None and crop.size > 0:
        mask_crop = mask[y1p:y2p, x1p:x2p]
        if mask_crop.shape[:2] == crop.shape[:2]:
            mask_bool = mask_crop.astype(bool)
            crop[~mask_bool] = (255, 255, 255)
    return crop


def filter_detections(boxes_xyxy, masks_data, roi_polygon, min_area, max_area):
    """
    IMPORTANT NOTE: if we have a mask, we compute the area from the
    ACTUAL PIXEL COUNT of the mask (not from the bbox).
    Mask area (count of TRUE pixels) is ROTATION-INVARIANT.
    """
    kept_centroids, kept_bboxes, kept_masks = [], [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if masks_data is not None:
            area = float(np.sum(masks_data[i] > 0.5))  # actual mask pixel area
        else:
            area = (x2 - x1) * (y2 - y1)  # fallback: bbox area 

        if not point_in_polygon((cx, cy), roi_polygon):
            continue
        if area < min_area or area > max_area:
            continue
        kept_centroids.append((cx, cy))
        kept_bboxes.append((x1, y1, x2, y2))
        kept_masks.append(masks_data[i] if masks_data is not None else None)
    return kept_centroids, kept_bboxes, kept_masks


# ======================================================================
# DINOv2 EMBEDDING 
# ======================================================================
def load_dinov2(device_override=None):
    """
    Uses Hugging Face's `transformers` library (not torch.hub), since
    torch.hub first has to open a GitHub page and occasionally hits a
    503/rate-limit error. transformers downloads directly from the
    Hugging Face Hub which is more reliable.

    We first try to load the model from a local cache (if available).
    If that fails, we fall back to downloading from the Hub.
    """
    from transformers import AutoImageProcessor, AutoModel
    import torch
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True)
        model = AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True)
    except Exception:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small")

    model.eval()
    model.to(device)
    return (model, processor), device


def embed_crop_dinov2(model_bundle, device, crop_bgr):
    import torch
    from PIL import Image

    model, processor = model_bundle
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    if getattr(outputs, "pooler_output", None) is not None:
        emb = outputs.pooler_output
    else:
        emb = outputs.last_hidden_state[:, 0]  # CLS token

    emb = emb.cpu().numpy().flatten()
    return emb / (np.linalg.norm(emb) + 1e-8)


# ======================================================================
# ONLINE CLUSTERER
# ======================================================================
class OnlineClusterer:
    """
    Naming logic for categories (Produktart N):
        - When an item doesn't match any existing category, a NEW
          category is created that has NO number yet (displayed as
          "Neue Kategorie").

        - When a category reaches its SECOND member,
          it gets a fixed number RIGHT AT THAT MOMENT, in
          chronological order: "Produktart 1", "Produktart 2", ...

        - Categories that still have only 1 member by the end of the
          video (never confirmed) are merged together in the final
          summary into "Unklar / Sonstige".
    """
    def __init__(self, similarity_threshold=0.75, max_thumbnails_per_cluster=8,
                 confirm_min_size=2):
        self.threshold = similarity_threshold
        self.clusters = []
        self.next_cluster_id = 0
        self.max_thumbs = max_thumbnails_per_cluster
        self.confirm_min_size = confirm_min_size
        self.confirmed_numbers = {}     # cluster_id -> 1, 2, 3, ... (in chronological confirmation order)
        self.next_confirmed_number = 1

    @staticmethod
    def _cosine_sim(a, b):
        a_n = a / (np.linalg.norm(a) + 1e-8)
        b_n = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_n, b_n))

    def add(self, embedding, thumbnail_bgr):
        embedding = np.array(embedding, dtype=float)
        if not self.clusters:
            self._create_cluster(embedding, thumbnail_bgr)
            return {"cluster_id": self.clusters[-1]["id"], "is_new": True, "similarity": None}

        sims = [self._cosine_sim(embedding, c["prototype"]) for c in self.clusters]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= self.threshold:
            self._assign(best_idx, embedding, thumbnail_bgr, best_sim)
            return {"cluster_id": self.clusters[best_idx]["id"], "is_new": False, "similarity": best_sim}
        else:
            self._create_cluster(embedding, thumbnail_bgr)
            # 'similarity' here means "how similar was it to the nearest EXISTING category"
            #  used for the terminal log message
            return {"cluster_id": self.clusters[-1]["id"], "is_new": True, "similarity": best_sim}

    def _create_cluster(self, embedding, thumbnail_bgr):
        self.clusters.append({"id": self.next_cluster_id, "prototype": embedding.copy(),
                               "count": 1, "thumbnails": [(thumbnail_bgr, None)]})
        self.next_cluster_id += 1

    def _assign(self, idx, embedding, thumbnail_bgr, sim):
        c = self.clusters[idx]
        n = c["count"]
        c["prototype"] = (c["prototype"] * n + embedding) / (n + 1)
        c["count"] = n + 1
        c["thumbnails"].append((thumbnail_bgr, sim))
        if len(c["thumbnails"]) > self.max_thumbs:
            c["thumbnails"] = c["thumbnails"][-self.max_thumbs:]

        # ---- Confirmation happens right here ----
        if c["count"] >= self.confirm_min_size and c["id"] not in self.confirmed_numbers:
            self.confirmed_numbers[c["id"]] = self.next_confirmed_number
            self.next_confirmed_number += 1

    def get_display_name(self, cluster_id):
        """Live label (while processing) -- always reflects the CURRENT state."""
        if cluster_id in self.confirmed_numbers:
            return f"Produktart {self.confirmed_numbers[cluster_id]}"
        return "Neue Kategorie"


def compute_final_summary(clusterer):
    """
    At the END of the video: confirmed categories keep
    their number; categories that were NEVER confirmed (only ever had
    1 member) are merged together -> "Unklar / Sonstige".
    """
    counts = {}
    for c in clusterer.clusters:
        if c["id"] in clusterer.confirmed_numbers:
            name = f"Produktart {clusterer.confirmed_numbers[c['id']]}"
        else:
            name = "Unklar / Sonstige"
        counts[name] = counts.get(name, 0) + c["count"]
    return counts


# ======================================================================
# TRACK + TRACKER
# ======================================================================
class Track:
    def __init__(self, track_id, centroid, bbox, mask, frame_idx):
        self.id = track_id
        self.centroid = np.array(centroid, dtype=float)
        self.bbox = bbox
        self.mask = mask
        self.age = 0
        self.counted = False
        self.count_number = None
        self.side = None
        self.consecutive_frames = 1

        area = bbox_area(bbox)
        self.min_area = area
        self.max_area = area
        self._area_sum = area
        self.history = [{"frame": frame_idx, "centroid": tuple(float(v) for v in self.centroid.tolist()),
                          "bbox": tuple(float(v) for v in bbox), "area": float(area)}]

        self.best_crop = None
        self.best_crop_area = 0
        self.cluster_id = None
        self.cluster_similarity = None
        self.multi_crops = []

    def average_area(self):
        return self._area_sum / len(self.history)

    def average_velocity(self, window=5):
        recent = self.history[-window:]
        if len(recent) < 2:
            return np.array([0.0, 0.0])
        diffs = []
        for i in range(1, len(recent)):
            c1 = np.array(recent[i - 1]["centroid"])
            c2 = np.array(recent[i]["centroid"])
            dt = max(recent[i]["frame"] - recent[i - 1]["frame"], 1)
            diffs.append((c2 - c1) / dt)
        return np.mean(diffs, axis=0)

    def predicted_centroid(self):
        v = self.average_velocity(window=5)
        return self.centroid + v * self.age

    def update_position(self, new_centroid, new_bbox, new_mask, frame_idx):
        self.centroid = np.array(new_centroid, dtype=float)
        self.bbox = new_bbox
        self.mask = new_mask
        self.age = 0
        self.consecutive_frames += 1
        area = bbox_area(new_bbox)
        self.min_area = min(self.min_area, area)
        self.max_area = max(self.max_area, area)
        self._area_sum += area
        self.history.append({"frame": frame_idx, "centroid": tuple(float(v) for v in self.centroid.tolist()),
                              "bbox": tuple(float(v) for v in new_bbox), "area": float(area)})


class CentroidTracker:
    def __init__(self, max_distance=80, max_age=15, motion_direction=(1.0, 0.0), line_ratio=0.5,
                 direction="increasing", hysteresis=5, stable_frames=10):
        """
        motion_direction:
            2D vector (dx, dy) representing the ACTUAL direction of
            belt motion, supports ANY angle (e.g. a diagonal belt). 
            Gets normalized internally.
        """
        self.tracks = {}
        self.archive = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_age = max_age
        v = np.array(motion_direction, dtype=float)
        self.motion_direction = v / (np.linalg.norm(v) + 1e-8)
        self.line_ratio = line_ratio
        self.direction = direction
        self.hysteresis = hysteresis
        self.stable_frames = stable_frames
        self.line_origin = None    # a point ON the line (for drawing)
        self.line_offset = None    # scalar projection of the line
        self.roi_origin = None     # reference point for computing projections
        self.newly_counted_ids = []
        self.count_so_far = 0

    def _set_line(self, roi_bbox_or_polygon):
        """
        roi_bbox_or_polygon: either (x1,y1,x2,y2) or an (N,2) point array.
        The counting line is PERPENDICULAR to motion_direction, positioned
        at line_ratio along the ROI's extent in the motion_direction axis.
        """
        if isinstance(roi_bbox_or_polygon, np.ndarray) and roi_bbox_or_polygon.ndim == 2:
            pts = roi_bbox_or_polygon
        else:
            x1, y1, x2, y2 = roi_bbox_or_polygon
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        self.roi_origin = pts.mean(axis=0)
        projections = [(p - self.roi_origin) @ self.motion_direction for p in pts]
        min_p, max_p = min(projections), max(projections)
        self.line_offset = min_p + (max_p - min_p) * self.line_ratio
        self.line_origin = self.roi_origin + self.motion_direction * self.line_offset

    def get_line_endpoints(self, length=1000):
        """Two endpoints for drawing the line (perpendicular to motion_direction), for cv2.line."""
        perp = np.array([-self.motion_direction[1], self.motion_direction[0]])
        p1 = self.line_origin + perp * length
        p2 = self.line_origin - perp * length
        return tuple(p1.astype(int)), tuple(p2.astype(int))

    def update(self, centroids, bboxes, masks, roi_bbox, frame_idx):
        if self.line_offset is None:
            self._set_line(roi_bbox)
        self.newly_counted_ids = []
        track_ids = list(self.tracks.keys())
        matched_tracks, matched_dets = set(), set()

        if track_ids and centroids:
            cost = np.zeros((len(track_ids), len(centroids)))
            for i, tid in enumerate(track_ids):
                predicted = self.tracks[tid].predicted_centroid()
                for j, c in enumerate(centroids):
                    cost[i, j] = np.linalg.norm(predicted - np.array(c))
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.max_distance:
                    tid = track_ids[r]
                    m = masks[c] if masks else None
                    self.tracks[tid].update_position(centroids[c], bboxes[c], m, frame_idx)
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].age += 1
                self.tracks[tid].consecutive_frames = 0

        for j in range(len(centroids)):
            if j not in matched_dets:
                m = masks[j] if masks else None
                new_track = Track(self.next_id, centroids[j], bboxes[j], m, frame_idx)
                self.tracks[self.next_id] = new_track
                self.archive[self.next_id] = new_track
                self.next_id += 1

        lo_t = self.line_offset - self.hysteresis
        hi_t = self.line_offset + self.hysteresis
        for tid, tr in self.tracks.items():
            if tr.counted:
                continue
            pos = (tr.centroid - self.roi_origin) @ self.motion_direction
            new_side = "low" if pos <= lo_t else ("high" if pos >= hi_t else None)
            if new_side is None:
                continue
            if tr.side is None:
                tr.side = new_side
                continue
            if new_side == tr.side:
                continue
            inc = (tr.side == "low" and new_side == "high")
            dec = (tr.side == "high" and new_side == "low")
            should_count = ((self.direction == "increasing" and inc) or
                             (self.direction == "decreasing" and dec) or
                             (self.direction == "any" and (inc or dec)))
            tr.side = new_side
            if should_count:
                tr.counted = True
                self.count_so_far += 1
                tr.count_number = self.count_so_far
                self.newly_counted_ids.append(tid)

        stale = [tid for tid, tr in self.tracks.items() if tr.age > self.max_age]
        for tid in stale:
            del self.tracks[tid]
        return self.tracks


# ======================================================================
# AUTOMATIC MOTION-DIRECTION DETECTION
# ======================================================================
def detect_belt_direction(model, cap_path, roi_polygon, conf, iou, imgsz, device,
                           min_area, max_area, sample_frames=150):
    """
    Returns: (dx, dy), the RAW motion-direction vector (NOT
    normalized, since only its direction matters).
    """
    cap = cv2.VideoCapture(cap_path)
    if not cap.isOpened():
        return (1.0, 0.0)

    roi_bbox = (float(roi_polygon[:, 0].min()), float(roi_polygon[:, 1].min()),
                float(roi_polygon[:, 0].max()), float(roi_polygon[:, 1].max()))

    probe_tracker = CentroidTracker(max_distance=100, max_age=30, motion_direction=(1.0, 0.0),
                                     direction="any", stable_frames=999999)

    frame_idx = 0
    while frame_idx < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        results = model(frame, device=device, retina_masks=False, imgsz=imgsz,
                         conf=conf, iou=iou, verbose=False)
        centroids, bboxes, masks_list = [], [], []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            centroids, bboxes, masks_list = filter_detections(
                boxes_xyxy, None, roi_polygon, min_area, max_area
            )
        probe_tracker.update(centroids, bboxes, None, roi_bbox, frame_idx)

    cap.release()

    velocities = []
    for tid, tr in probe_tracker.archive.items():
        if len(tr.history) >= 3:
            velocities.append(tr.average_velocity(window=len(tr.history)))

    if not velocities:
        return (1.0, 0.0)

    avg_v = np.mean(velocities, axis=0)
    norm = np.linalg.norm(avg_v)
    if norm < 1e-6:
        return (1.0, 0.0)
    return (float(avg_v[0]), float(avg_v[1]))


# ======================================================================
# STATS COLUMN
# ======================================================================
def build_stats_column(tracker, clusterer, frame_height, width=260):
    """
    Live column: each category is shown with its CURRENT name.
    """
    col = np.full((frame_height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(col, (0, 0), (width - 1, frame_height - 1), (180, 180, 180), 1)
    cv2.putText(col, "Ergebnisse (live)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.line(col, (10, 42), (width - 10, 42), (200, 200, 200), 1)
    cv2.putText(col, f"Gesamt: {tracker.count_so_far}", (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    y = 100
    for c in sorted(clusterer.clusters, key=lambda c: -c["count"]):
        name = clusterer.get_display_name(c["id"])
        cv2.putText(col, f"{name}: {c['count']}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1)
        y += 26
        if y > frame_height - 20:
            break
    return col


# ======================================================================
# ONE COMPLETE PASS OVER THE VIDEO
# ======================================================================
def _process_video(
    input_video, model, dinov2_model, dinov2_device,
    conf, iou, imgsz, device,
    max_distance, max_age, motion_direction, line_ratio, direction, hysteresis, stable_frames,
    roi_polygon, min_area_val, max_area_val,
    cluster_threshold, confirm_min_size,
    average_embeddings, avg_embedding_interval, avg_embedding_max_samples,
    write_video_path=None, show_live=False,
    progress_callback=None,
):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi_bbox = (float(roi_polygon[:, 0].min()), float(roi_polygon[:, 1].min()),
                float(roi_polygon[:, 0].max()), float(roi_polygon[:, 1].max()))

    tracker = CentroidTracker(max_distance=max_distance, max_age=max_age, motion_direction=motion_direction,
                               line_ratio=line_ratio, direction=direction, hysteresis=hysteresis,
                               stable_frames=stable_frames)
    clusterer = OnlineClusterer(similarity_threshold=cluster_threshold, confirm_min_size=confirm_min_size)

    writer = None
    stats_col_width = 260
    if write_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(write_video_path, fourcc, fps, (frame_w + stats_col_width, frame_h))

    if show_live:
        cv2.namedWindow("BakeryDemo", cv2.WINDOW_NORMAL)

    COLOR_YELLOW, COLOR_BLUE, COLOR_GREEN = (0, 255, 255), (255, 120, 0), (0, 255, 0)
    flash_counter, FLASH_DURATION = 0, 6
    processing_fps = 0.0
    last_frame_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        now = time.time()
        instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
        # exponential smoothing to prevent the number from jittering
        processing_fps = (processing_fps * 0.8 + instant_fps * 0.2) if processing_fps > 0 else instant_fps
        last_frame_time = now

        results = model(frame, device=device, retina_masks=True, imgsz=imgsz,
                         conf=conf, iou=iou, verbose=False)
        centroids, bboxes, masks_list = [], [], []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            masks_data = r.masks.data.cpu().numpy() if r.masks is not None else None
            centroids, bboxes, masks_list = filter_detections(
                boxes_xyxy, masks_data, roi_polygon, min_area_val, max_area_val
            )

        tracks = tracker.update(centroids, bboxes, masks_list if masks_list else None, roi_bbox, frame_idx)
        if tracker.newly_counted_ids:
            flash_counter = FLASH_DURATION

        for tid, tr in tracks.items():
            area = bbox_area(tr.bbox)
            if area > tr.best_crop_area:
                crop_img = extract_masked_crop(frame, tr.bbox, tr.mask)
                if crop_img.size > 0:
                    tr.best_crop, tr.best_crop_area = crop_img, area
            if (average_embeddings and not tr.counted
                    and tr.consecutive_frames % avg_embedding_interval == 0
                    and len(tr.multi_crops) < avg_embedding_max_samples):
                sample_crop = extract_masked_crop(frame, tr.bbox, tr.mask)
                if sample_crop.size > 0:
                    tr.multi_crops.append(sample_crop)

        for tid in tracker.newly_counted_ids:
            tr = tracker.tracks.get(tid) or tracker.archive.get(tid)
            if tr is None:
                continue
            final_crop = extract_masked_crop(frame, tr.bbox, tr.mask)
            if final_crop.size == 0:
                continue
            if average_embeddings:
                crops_to_embed = tr.multi_crops + [final_crop]
                embs = [embed_crop_dinov2(dinov2_model, dinov2_device, c) for c in crops_to_embed]
                avg = np.mean(embs, axis=0)
                embedding = avg / (np.linalg.norm(avg) + 1e-8)
            else:
                embedding = embed_crop_dinov2(dinov2_model, dinov2_device, final_crop)
            result = clusterer.add(embedding, final_crop)
            tr.cluster_id = result["cluster_id"]
            tr.cluster_similarity = None if result["is_new"] else result["similarity"]

            # Terminal debug log
            if result["is_new"]:
                if result["similarity"] is not None:
                    print(f"[NEW CATEGORY] Track-ID={tid} (Count #{tr.count_number}): "
                          f"does not match ANY existing category "
                          f"(highest similarity: {result['similarity']*100:.1f}%, "
                          f"threshold: {cluster_threshold*100:.0f}%) "
                          f"-> new category created.")
                else:
                    print(f"[NEW CATEGORY] Track-ID={tid} (Count #{tr.count_number}): "
                          f"first category overall -> created.")

        if writer is not None or show_live:
            cv2.polylines(frame, [roi_polygon.astype(np.int32)], True, (0, 255, 255), 1)

            line_color = (255, 255, 255) if flash_counter > 0 else (0, 0, 255)
            line_thick = 4 if flash_counter > 0 else 2
            p1, p2 = tracker.get_line_endpoints(length=max(frame_w, frame_h))
            cv2.line(frame, p1, p2, line_color, line_thick)
            if flash_counter > 0:
                flash_counter -= 1

            for tid, tr in tracks.items():
                if tr.counted:
                    # NOTE: OpenCV's built-in Hershey fonts (cv2.putText)
                    # do not reliably render German umlauts (ä/ö/ü/ß) --
                    # they can come out corrupted or missing. This text
                    # is baked directly into the video, so it stays in
                    # the safe ASCII form here.
                    color, state_label = COLOR_GREEN, "GEZAEHLT"
                elif tr.consecutive_frames >= stable_frames:
                    color, state_label = COLOR_BLUE, "STABIL"
                else:
                    color, state_label = COLOR_YELLOW, "NEU"

                x1, y1, x2, y2 = map(int, tr.bbox)
                if tr.mask is not None:
                    mask_bool = tr.mask.astype(bool)
                    overlay = frame.copy()
                    overlay[mask_bool] = color
                    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {tid} [{state_label}]"
                if tr.counted:
                    label += f" #{tr.count_number}"
                    if tr.cluster_id is not None:
                        label += f" {clusterer.get_display_name(tr.cluster_id)}"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (int(tr.centroid[0]), int(tr.centroid[1])), 4, color, -1)

            cv2.putText(frame, f"Frame {frame_idx}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            stats_col = build_stats_column(tracker, clusterer, frame_h, width=260)
            combined = cv2.hconcat([frame, stats_col])

            if writer is not None:
                writer.write(combined)
            if show_live:
                cv2.imshow("BakeryDemo", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        if progress_callback is not None and total_frames > 0:
            progress_callback(frame_idx, total_frames, fps=processing_fps, device=device)

    cap.release()
    if writer is not None:
        writer.release()
    if show_live:
        cv2.destroyAllWindows()

    return tracker, clusterer


# ======================================================================
# RUN_PIPELINE
# ======================================================================
def run_pipeline(
    input_video,
    output_dir="output",
    fastsam_model_name="FastSAM-s.pt",
    conf=0.5, iou=0.9, imgsz=640, device="cpu",
    max_distance=80, max_age=25,
    belt_angle_deg="auto", line_ratio=0.5, direction="increasing", hysteresis=5,
    stable_frames=10,
    roi_points=None,
    min_area=None, max_area=None, min_area_ratio=0.001, max_area_ratio=0.20,
    cluster_threshold=0.75, cluster_device=None, confirm_min_size=2,
    average_embeddings=True, avg_embedding_interval=5, avg_embedding_max_samples=5,
    show_live=False,
    progress_callback=None,
):
    """
    belt_angle_deg:
        "auto"  -> the actual motion direction is automatically detected
                   from the video itself and supports ANY angle.

        float   -> a MANUAL angle in degrees (0 = right/+x, 90 = down/+y,
                   45 = diagonal, ...). `direction` still determines
                   which crossing side is counted.
    """
    from ultralytics import FastSAM
    import math

    os.makedirs(output_dir, exist_ok=True)
    clusters_dir = os.path.join(output_dir, "clusters")
    if os.path.exists(clusters_dir):
        shutil.rmtree(clusters_dir)
    os.makedirs(clusters_dir, exist_ok=True)

    model = FastSAM(fastsam_model_name)
    dinov2_model, dinov2_device = load_dinov2(device_override=cluster_device)

    cap = cv2.VideoCapture(input_video)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if roi_points:
        pts = roi_points
        roi_polygon = np.array([[pts[0], pts[1]], [pts[2], pts[3]],
                                 [pts[4], pts[5]], [pts[6], pts[7]]], dtype=np.float32)
    else:
        roi_polygon = np.array([[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]], dtype=np.float32)

    roi_area_val = polygon_area(roi_polygon)
    min_area_val = min_area if min_area is not None else roi_area_val * min_area_ratio
    max_area_val = max_area if max_area is not None else roi_area_val * max_area_ratio

    # ---- Automatic direction detection: this is a SEPARATE, SHORT pass
    # only over the first few hundred frames.

    if belt_angle_deg == "auto":
        if progress_callback:
            progress_callback(0, 1, phase="Erkenne Förderband-Richtung...")
        motion_direction = detect_belt_direction(
            model, input_video, roi_polygon, conf, iou, imgsz, device, min_area_val, max_area_val,
        )
        detected_angle = math.degrees(math.atan2(motion_direction[1], motion_direction[0]))
    else:
        rad = math.radians(float(belt_angle_deg))
        motion_direction = (math.cos(rad), math.sin(rad))
        detected_angle = float(belt_angle_deg)

    def main_progress(cur, total, fps=None, device=None):
        if progress_callback:
            progress_callback(cur, total, phase="Verarbeitung...", fps=fps, device=device)

    video_path = os.path.join(output_dir, "demo_video.mp4")
    tracker, clusterer = _process_video(
        input_video, model, dinov2_model, dinov2_device,
        conf, iou, imgsz, device, max_distance, max_age, motion_direction, line_ratio,
        direction, hysteresis, stable_frames, roi_polygon, min_area_val, max_area_val,
        cluster_threshold, confirm_min_size,
        average_embeddings, avg_embedding_interval, avg_embedding_max_samples,
        write_video_path=video_path, show_live=show_live,
        progress_callback=main_progress,
    )

    # ================================================================
    # FINAL OUTPUT
    # ================================================================
    final_counts_by_cluster_id = {}
    for c in clusterer.clusters:
        name = (f"Produktart {clusterer.confirmed_numbers[c['id']]}"
                if c["id"] in clusterer.confirmed_numbers else "Unklar / Sonstige")
        final_counts_by_cluster_id[c["id"]] = name

    for name in set(final_counts_by_cluster_id.values()):
        folder_name = name.replace(" / ", "_").replace(" ", "_")
        os.makedirs(os.path.join(clusters_dir, folder_name), exist_ok=True)

    clusters_json = {name: [] for name in set(final_counts_by_cluster_id.values())}
    counts = {}
    for tid, tr in tracker.archive.items():
        if not tr.counted or tr.cluster_id is None:
            continue
        name = final_counts_by_cluster_id.get(tr.cluster_id, "Unklar / Sonstige")
        folder_name = name.replace(" / ", "_").replace(" ", "_")
        folder_path = os.path.join(clusters_dir, folder_name)

        sim = tr.cluster_similarity
        sim_pct = int(round(sim * 100)) if sim is not None else 100
        img_filename = f"track_{tid:04d}_count{tr.count_number:03d}_sim{sim_pct}.png"
        img_path = os.path.join(folder_path, img_filename)
        if tr.best_crop is not None:
            cv2.imwrite(img_path, tr.best_crop)

        clusters_json[name].append({
            "bild": os.path.relpath(img_path, output_dir),
            "ähnlichkeit_prozent": sim_pct,
            "track_id": tid,
            "zähl_nummer": tr.count_number,
        })
        counts[name] = counts.get(name, 0) + 1

    gesamt = sum(counts.values())
    results = {"gesamt": gesamt, "anzahl": counts, "produktarten": clusters_json,
               "erkannte_förderband_richtung": {"winkel_grad": round(detected_angle, 1), "richtung": direction}}

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Produktart,Anzahl\n")
        for name, cnt in counts.items():
            f.write(f"{name},{cnt}\n")
        f.write(f"Gesamt,{gesamt}\n")

    # Downloadable ZIP archive of the per-category crop folders.
    zip_base = os.path.join(output_dir, "produktarten_bilder")
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=clusters_dir)

    return {
        "gesamt": gesamt, "counts": counts,
        "video_path": video_path, "json_path": json_path, "csv_path": csv_path,
        "clusters_dir": clusters_dir, "zip_path": zip_path,
        "detected_angle_deg": round(detected_angle, 1), "detected_direction": direction,
    }
