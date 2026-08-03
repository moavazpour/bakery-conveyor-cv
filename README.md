# Real-Time Detection, Tracking & Classification on a Moving Conveyor

![Demo](assets/demo_video_2.gif)

An end-to-end computer-vision pipeline that segments, tracks, counts and classifies baked goods on a moving conveyor belt — **without a single labeled training image**.

I built this as an independent project (08/2026) to explore how far modern foundation models can take an industrial inspection task when there is no annotated dataset and no time to train a custom detector. The answer: pretty far — FastSAM handles segmentation, DINOv2 handles appearance, and a small amount of classical tracking and state-machine logic turns that into reliable counts.

**[Try the live demo on Hugging Face](https://huggingface.co/spaces/moavazpour/bakeryDemo)**

## Highlights

- **FastSAM** with ROI-relative area filtering, which separates touching or overlapping objects.
- **Hungarian data association** with velocity-based prediction to bridge short detection dropouts.
- **Four-stage line-crossing state machine** with hysteresis; each track is counted once.
- **Online classification** with DINOv2 embeddings and incremental clustering; unrepeated noise is routed to an "unclear" class.
- **Outputs:** CLI, annotated video, JSON/CSV export, sorted image archive; Gradio demo deployed on Hugging Face.

**Stack:** Python, FastSAM, DINOv2, OpenCV, Hungarian algorithm, Gradio, Hugging Face

## Project Structure

```
src/
├── pipeline.py         Core logic (detection, tracking, counting, clustering) -- HEADLESS
├── run_local.py        Local execution, saves video + JSON/CSV/ZIP, optional live preview
└── app.py              Web demo (Hugging Face Spaces)
```

## Quick Start

```bash
conda create -n bakery-cv python=3.10 -y
conda activate bakery-cv
pip install -r requirements.txt
```

```bash
# Process a video and save the results to disk
python src/run_local.py --input data/input_video.mp4

# With a live preview during processing (in addition to saving)
python src/run_local.py --input data/input_video.mp4 --live
```

The belt direction is **detected automatically** by default (any angle, including
diagonal belts). It can also be set manually:

```bash
python src/run_local.py --input data/input_video.mp4 --belt-angle-deg 45 --direction increasing
```

To restrict processing to part of the frame, pass a Region of Interest (ROI) as 4 points
(8 values) via `--roi-points`.

## Web Demo

No local installation needed:

1. Open the **[Hugging Face Space](https://huggingface.co/spaces/moavazpour/bakeryDemo)**.
2. Upload a conveyor belt video.
3. Click **Analyse starten** (Start analysis) and wait until processing is finished.
4. Watch the annotated output video (bounding boxes, counting line, live statistics) and download the results (CSV/JSON/ZIP).

## Output

```
output/
├── demo_video.mp4              Video with bounding boxes, counting line and a live statistics panel
├── results.json                Total count + list of all images per product type
├── results.csv
└── produktarten_bilder.zip     Downloadable ZIP of all images, sorted by product type
    └── clusters/
        ├── Produktart_1/           Cropped images ("product type 1"); file names include the similarity in %
        ├── Produktart_2/
        └── Unklar_Sonstige/        Unclear / other
```

---

## How It Works

```
Video frame ─► FastSAM masks ─► Area filter ─► Hungarian tracking ─► Line-crossing ─► Count
                                                       │                state machine
                                                       └──► DINOv2 embedding ─► Online clustering ─► Product type
```

### 1. Segmentation: why FastSAM

The obvious first idea was background subtraction, but it breaks down when the
camera is zoomed in tightly: the belt is almost never visible without an object
on it, so there is no clean empty background to learn from.

Since I wanted the pipeline to work without any labeled data or custom training,
I chose **FastSAM**. It is pre-trained on millions of generic objects and
produces a **separate, precise mask for every single object** rather than one
coarse "baked goods" blob — which matters most exactly when items touch or
slightly overlap. I use the small **FastSAM-s** variant to keep processing time
low.

To reject false detections (e.g. parts of the belt itself), masks are filtered
by minimum and maximum area. The limits are defined as a **fraction of the ROI
area** rather than in pixels, so they work regardless of camera resolution and
zoom.

### 2. Tracking: Hungarian matching with motion prediction

To count each item once, objects have to be followed across frames. Something
like DeepSORT would be overkill here. Instead I use the **Hungarian algorithm**:
it looks at the full distance matrix between previous and current positions and
picks the assignment that **minimises the total distance**, which avoids
mix-ups when items are close together. As a classical optimisation method, it
costs almost nothing compared to the neural networks in the pipeline.

Masks occasionally drop out for a frame or two. Because the belt moves at a
predictable speed, the pipeline estimates the **average velocity** of all
tracked objects and predicts where each one should be, so a briefly missed item
is re-associated with its existing track instead of getting a new ID. A track
is only discarded after a short grace period without detections.

### 3. Counting: a line-crossing state machine

"Count every track ID" sounds simple, but short-lived noise detections also
create IDs and would inflate the count. Instead, counting is done with a
virtual line and a four-part state machine:

1. **Vector projection instead of fixed x/y coordinates:** the object position
   is projected onto the actual direction of motion
   (`pos = (center - roi_origin) · direction_of_motion`), so the logic works for
   belts at any angle.
2. **State instead of snapshot:** each object is either "before" or "after" the
   line; a count happens only on a real state change, not because an object
   happens to first appear on the far side.
3. **Hysteresis zone:** a small buffer around the line stops mask jitter from
   flipping the state back and forth and producing double counts.
4. **Counted once, counted forever:** once an object is counted, it is
   permanently excluded from further checks.

### 4. Classification: DINOv2 + online clustering

Each cropped item is turned into a feature vector with **DINOv2**, a
self-supervised Vision Transformer trained on ~142M images (LVD-142M). I chose
it over hand-crafted features (colour histograms, simple shape descriptors)
because those are far more sensitive to viewing angle, lighting and colour
shifts.

Items need a category the moment they appear, so clustering runs **online**
rather than as a batch step at the end (as HDBSCAN would). A new object starts a
provisional category; later objects with similar embeddings join it. A category
is confirmed and numbered once it gets a second member. Product types on a
production line repeat, while noise usually doesn't — so categories that never
get a second member are merged into **"Unklar / Sonstige"** (unclear / other)
at the end.

---

## Assumptions

- The camera position and viewing angle stay constant.
- The camera looks straight down onto the belt.
- Lighting is adequate and even.
- The full belt width is visible in the frame.
- Products overlap by at most ~30% of their volume.
- The resolution is high enough to see product boundaries and details clearly —
  on low-quality footage, separating individual items gets noticeably harder for
  both detection and clustering.
- The viewing angle is not so shallow that boundaries get lost to occlusion or
  perspective.
- Frame rate, shutter speed, lighting and compute are matched to the belt speed.

## Limitations

- On a real production line, the thresholds (especially for mask detection)
  would need an initial calibration, ideally fine-tuning, for best results.
- Heavy overlap makes separation much harder and hurts both counting and
  classification. Where heavy overlap is unavoidable, a supervised model could
  separate items better — as long as each item is still visible to the camera.
- Online clustering can occasionally mis-assign items; by design it is somewhat
  less accurate than batch methods like HDBSCAN that see the whole dataset at
  once.

## How I Validated It

- **Single-product videos:** checked that no extra categories appear; when they
  did, I traced the cause in the terminal output (similarity to the nearest
  category vs. the threshold).
- **Mixed-product videos:** checked that items grouped together actually belong
  together, and that unrelated items aren't mixed.
- **Category separation:** visually checked that different categories really
  are different products, and that similar products weren't split unnecessarily.
- **Reference count:** ran a video with a known, externally determined count
  through the pipeline and analysed the deviations.
- **"Unklar / Sonstige" review:** confirmed that items there genuinely didn't fit
  any other category.
- **Manual spot counts:** manually counted sections of video and compared them
  with the automatic total.
- **Tracking stability:** watched for mask flicker and lost or switching track IDs.
- **Reproducibility:** ran the same video several times to confirm the
  clustering result stays consistent.

## Parameters

| Parameter | Meaning |
|---|---|
| `--input` | Path to the input video (required) |
| `--output-dir` | Target folder for all outputs |
| `--model` | FastSAM model size (default: `FastSAM-s.pt`) |
| `--conf` | Detection confidence threshold |
| `--iou` | NMS threshold (prevents duplicate, overlapping detections) |
| `--imgsz` | Image size FastSAM rescales to internally |
| `--device` / `--cluster-device` | CPU or GPU, selectable independently for FastSAM and DINOv2 |
| `--max-distance` | Maximum pixel distance for a valid frame-to-frame match (Hungarian matching) |
| `--max-age` | How many frames an object may be "missing" before its track is dropped |
| `--belt-angle-deg` | `auto` (automatic direction detection) or a fixed angle in degrees |
| `--direction` | Which crossing direction of the counting line is counted |
| `--line-ratio` | Position of the counting line along the direction of motion |
| `--hysteresis` | Width of the buffer zone around the counting line to suppress jitter |
| `--stable-frames` | Frames until an object is marked as "stably tracked" |
| `--roi-points` | 4 points (8 values) defining the relevant image region |
| `--min-area-ratio` / `--max-area-ratio` | Allowed object area relative to the ROI area |
| `--cluster-threshold` | Cosine similarity threshold for categorisation |
| `--confirm-min-size` | Minimum number of members before a category gets a fixed number |
| `--avg-embedding-interval` / `--avg-embedding-max-samples` | Control of feature averaging across multiple views |
| `--live` | Additional live preview (local only) |
