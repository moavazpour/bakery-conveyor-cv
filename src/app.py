"""
app.py (Hugging Face Space entry point)
==========================================
This code Runs a web-based demo for the Bakery Förderband (conveyor belt) detection, tracking, and counting system.

License: MIT
"""

import os
import shutil
import tempfile
import time
import cv2
import gradio as gr
from pipeline import run_pipeline

try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    # Not running on a Hugging Face ZeroGPU Space (e.g. local machine) --
    # just call the function directly, no GPU allocation wrapper needed.
    def gpu_decorator(func):
        return func


# ======================================================================
# SECTION 1: ROI PICKER (click 4 points on the first frame)
# ======================================================================
def extract_first_frame(video_path):
    if video_path is None:
        raise gr.Error("Bitte zuerst ein Video hochladen.")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise gr.Error("Der erste Frame konnte nicht gelesen werden.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, frame_rgb, []


def on_image_click(evt: gr.SelectData, original_frame, points):
    if original_frame is None:
        raise gr.Error("Bitte zuerst auf 'Ersten Frame anzeigen' klicken.")
    points = list(points) if points else []
    if len(points) >= 4:
        gr.Warning("Es sind bereits 4 Punkte gesetzt. Bitte zuerst 'ROI zurücksetzen' klicken.")
        return original_frame, points

    x, y = evt.index[0], evt.index[1]
    points.append((x, y))

    display = original_frame.copy()
    for i, (px, py) in enumerate(points):
        cv2.circle(display, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(display, str(i + 1), (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(display, points[i], points[i + 1], (255, 255, 0), 2)
    if len(points) == 4:
        cv2.line(display, points[3], points[0], (255, 255, 0), 2)

    return display, points


def reset_roi(original_frame):
    return original_frame, []


def points_to_roi_string(points):
    if len(points) != 4:
        return "Kein ROI gesetzt -- das gesamte Bild wird verwendet."
    flat = " ".join(str(v) for pt in points for v in pt)
    return f"ROI gesetzt: {flat}"


# ======================================================================
# SECTION 2: MAIN PROCESSING
# ======================================================================
@gpu_decorator
def process_video(
    video_path,
    roi_points_state,
    conf,
    min_area_ratio,
    max_area_ratio,
    cluster_threshold,
    belt_angle_deg,
    direction,
    line_ratio,
    stable_frames,
    confirm_min_size,
    average_embeddings,
    progress=gr.Progress(),
):
    if video_path is None:
        raise gr.Error("Bitte zuerst ein Video hochladen.")

    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Only on Windows: make an independent copy so our processing doesn't
    # compete with Gradio's own file server for read access.
    tmp_dir = tempfile.mkdtemp(prefix="bakery_cv_")
    local_video_path = os.path.join(tmp_dir, "input_video.mp4")
    shutil.copy(video_path, local_video_path)

    # Unique output folder per run.
    run_output_dir = os.path.join("output_gradio", f"run_{int(time.time() * 1000)}")

    angle_arg = "auto" if belt_angle_deg == "auto" else float(belt_angle_deg)

    roi_points_flat = None
    if roi_points_state and len(roi_points_state) == 4:
        roi_points_flat = [float(v) for pt in roi_points_state for v in pt]

    def progress_cb(current, total, phase="", fps=None, device=None):
        frac = (current / total) if total else 0
        if fps is not None:
            gpu_text = "GPU" if device == "cuda" else "CPU"
            desc = f"{phase} Frame {current}/{total} | {gpu_text} | {fps:.1f} Bilder/Sekunde"
        else:
            desc = f"{phase} {current}/{total}"
        progress(frac, desc=desc)

    progress(0, desc="Starte Verarbeitung...")
    try:
        summary = run_pipeline(
            input_video=local_video_path,
            output_dir=run_output_dir,
            device=device_str,
            cluster_device=device_str,
            conf=conf,
            roi_points=roi_points_flat,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            cluster_threshold=cluster_threshold,
            belt_angle_deg=angle_arg,
            direction=direction,
            line_ratio=line_ratio,
            stable_frames=stable_frames,
            confirm_min_size=confirm_min_size,
            average_embeddings=average_embeddings,
            show_live=False,
            progress_callback=progress_cb,
        )
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise gr.Error(f"Fehler während der Verarbeitung: {e}")

    progress(0.99, desc="Speichere Ergebnisse und erstelle ZIP-Datei...")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    # Also remove the original uploaded file, since we don't need it anymore.

    try:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
    except OSError:
        pass

    summary_text = (
        f"### Fertig!\n\n"
        f"**Förderband erkannt:** Winkel={summary['detected_angle_deg']} Grad, "
        f"Richtung={summary['detected_direction']}\n\n"
        f"### Gesamt: {summary['gesamt']}\n\n"
    )
    for name, count in summary["counts"].items():
        summary_text += f"- **{name}**: {count}\n"

    progress(1.0, desc="Fertig!")
    return (
        summary_text,
        summary["video_path"], summary["json_path"], summary["csv_path"], summary["zip_path"],
    )


def set_busy():
    return gr.update(interactive=False, value="Wird verarbeitet... bitte warten")


def set_idle():
    return gr.update(interactive=True, value="Analyse starten")


# ======================================================================
# SECTION 3: UI LAYOUT
# ======================================================================
with gr.Blocks(title="Bakery Förderband — Erkennung, Tracking & Zählung") as demo:
    gr.Markdown(
        "# Bakery Förderband: Erkennung, Tracking & Zählung\n"
        "Ein Video eines Förderbands hochladen. Das System erkennt, verfolgt und "
        "zählt die Backwaren automatisch und leitet die Produktarten selbstständig "
        "aus dem Videomaterial ab, ohne vortrainiertes Modell und ohne Datensatz. "
        "Eine verfügbare GPU wird automatisch genutzt."
    )

    roi_points_state = gr.State([])
    raw_frame_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Eingabevideo (Förderband)")

            with gr.Accordion("Bereich (ROI) festlegen (optional)", open=False):
                gr.Markdown(
                    "Zuerst den ersten Frame anzeigen, dann **4 Punkte anklicken** "
                    "(die Ecken des Bereichs, der das Förderband umschließt). "
                    "Ohne Festlegung wird das gesamte Bild verwendet."
                )
                show_frame_btn = gr.Button("Ersten Frame anzeigen")
                roi_image = gr.Image(label="Auf 4 Punkte klicken (Reihenfolge: im Kreis herum)",
                                      interactive=False)
                roi_status = gr.Markdown("Kein ROI gesetzt.")
                reset_roi_btn = gr.Button("ROI zurücksetzen")

            with gr.Accordion("Erweiterte Einstellungen", open=False):
                conf = gr.Slider(
                    0.1, 0.9, value=0.5, step=0.05, label="Konfidenz-Schwelle (Detection)",
                    info="Mindest-Sicherheit für eine Erkennung. Höher = strenger, weniger "
                         "Fehlerkennungen, aber möglicherweise werden Objekte übersehen.")
                min_area_ratio = gr.Slider(
                    0.0001, 0.05, value=0.001, step=0.0001, label="Mindestfläche (relativ zum ROI)",
                    info="Kleinste erlaubte Objektfläche als Anteil der ROI-Fläche. "
                         "Filtert kleine Bildstörungen (Rauschen) heraus.")
                max_area_ratio = gr.Slider(
                    0.05, 0.9, value=0.20, step=0.01, label="Maximalfläche (relativ zum ROI)",
                    info="Größte erlaubte Objektfläche als Anteil der ROI-Fläche. "
                         "Verhindert, dass das Förderband selbst fälschlich als Objekt erkannt wird.")
                cluster_threshold = gr.Slider(
                    0.5, 0.95, value=0.75, step=0.01, label="Ähnlichkeits-Schwelle (Kategorisierung)",
                    info="Ab dieser Kosinus-Ähnlichkeit wird ein Objekt einer bestehenden "
                         "Produktart zugeordnet. Höher = mehr, feiner unterschiedene Produktarten.")
                belt_angle_deg = gr.Textbox(
                    value="auto", label="Förderband-Winkel (Grad) oder 'auto'",
                    info="'auto' erkennt den Winkel automatisch aus der tatsächlichen Bewegung "
                         "im Video. Alternativ ein fester Winkel in Grad: 0=nach rechts, "
                         "90=nach unten, 45=diagonal.")
                direction = gr.Dropdown(
                    ["increasing", "decreasing", "any"], value="increasing",
                    label="Zählrichtung",
                    info="Welche Überquerungsrichtung der Zähllinie gezählt wird "
                         "('increasing' = in Bewegungsrichtung, 'any' = beide Richtungen).")
                line_ratio = gr.Slider(
                    0.1, 0.9, value=0.5, step=0.05, label="Position der Zähllinie",
                    info="Position der Zähllinie entlang der Bewegungsrichtung, als Anteil "
                         "der ROI-Ausdehnung (0.5 = Mitte).")
                stable_frames = gr.Slider(
                    3, 30, value=10, step=1, label="Frames bis 'stabil'",
                    info="Anzahl aufeinanderfolgender Frames, nach denen ein Objekt als "
                         "'stabil verfolgt' (blau) statt 'neu' (gelb) markiert wird.")
                confirm_min_size = gr.Slider(
                    1, 10, value=2, step=1, label="Mindestgröße für gültige Produktart",
                    info="Ab dieser Mitgliederzahl erhält eine Kategorie eine feste Nummer "
                         "('Produktart N'). Darunter wird sie am Ende zu 'Unklar / Sonstige'.")
                average_embeddings = gr.Checkbox(
                    value=True, label="Merkmal-Mittelung über mehrere Ansichten",
                    info="Mittelt das Erkennungsmerkmal über mehrere Ansichten desselben "
                         "Objekts während der Verfolgung -- macht die Kategorisierung robuster "
                         "gegenüber Blickwinkel- und Beleuchtungsschwankungen.")

            run_btn = gr.Button("Analyse starten", variant="primary")

        with gr.Column(scale=1):
            summary_output = gr.Markdown()
            video_output = gr.Video(label="Ergebnis")
            with gr.Row():
                json_output = gr.File(label="results.json")
                csv_output = gr.File(label="results.csv")
            zip_output = gr.File(label="Bilder pro Produktart (ZIP)")

    show_frame_btn.click(
        fn=extract_first_frame,
        inputs=[video_input],
        outputs=[roi_image, raw_frame_state, roi_points_state],
    )
    roi_image.select(
        fn=on_image_click,
        inputs=[raw_frame_state, roi_points_state],
        outputs=[roi_image, roi_points_state],
    ).then(
        fn=points_to_roi_string,
        inputs=[roi_points_state],
        outputs=[roi_status],
    )
    reset_roi_btn.click(
        fn=reset_roi,
        inputs=[raw_frame_state],
        outputs=[roi_image, roi_points_state],
    ).then(
        fn=lambda: "Kein ROI gesetzt.",
        outputs=[roi_status],
    )

    # Busy-state chain: grey out and relabel the button while processing,

    run_btn.click(
        fn=set_busy, outputs=[run_btn],
    ).then(
        fn=process_video,
        inputs=[video_input, roi_points_state, conf, min_area_ratio, max_area_ratio, cluster_threshold,
                belt_angle_deg, direction, line_ratio, stable_frames, confirm_min_size,
                average_embeddings],
        outputs=[summary_output, video_output, json_output, csv_output, zip_output],
        show_progress_on=[video_output],
    ).then(
        fn=set_idle, outputs=[run_btn],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()