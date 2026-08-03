"""
run_local.py
=============
Local execution of the pipeline. The final video is ALWAYS saved to
disk (output/demo_video.mp4). If you pass --live, a live preview window
also opens simultaneously, showing exactly the same frames that are
being written to the file.

Simple run (no live view, just save):
    python run_local.py --input data/input_video.mp4

Run with live preview (+ simultaneous saving):
    python run_local.py --input data/input_video.mp4 --live

The conveyor belt's motion direction is detected AUTOMATICALLY by
default. To set it manually instead:
    python run_local.py --input data/input_video.mp4 --belt-angle-deg 45 --direction increasing
"""

import argparse
from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model", default="FastSAM-s.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="FastSAM device: cpu or cuda")
    parser.add_argument("--max-distance", type=int, default=80)
    parser.add_argument("--max-age", type=int, default=25)

    parser.add_argument("--belt-angle-deg", default="auto",
                         help="'auto' = automatic detection (any angle, not just x/y). "
                              "Or a fixed angle in degrees: 0=right, 90=down, 45=diagonal, etc.")
    parser.add_argument("--direction", choices=["increasing", "decreasing", "any"], default="increasing")
    parser.add_argument("--line-ratio", type=float, default=0.5)
    parser.add_argument("--hysteresis", type=int, default=5)
    parser.add_argument("--stable-frames", type=int, default=10)

    parser.add_argument("--roi-points", type=float, nargs=8, default=None,
                         metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"))
    parser.add_argument("--min-area", type=float, default=None)
    parser.add_argument("--max-area", type=float, default=None)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.20)

    parser.add_argument("--cluster-threshold", type=float, default=0.75)
    parser.add_argument("--cluster-device", default=None,
                         help="DINOv2 device for clustering: cpu, cuda, or leave empty "
                              "for automatic selection")
    parser.add_argument("--confirm-min-size", type=int, default=2,
                         help="Once a category reaches this many members, it officially "
                              "gets a number (Produktart N); otherwise it ends up as Unklar")
    parser.add_argument("--no-average-embeddings", action="store_true")
    parser.add_argument("--avg-embedding-interval", type=int, default=5)
    parser.add_argument("--avg-embedding-max-samples", type=int, default=5)

    parser.add_argument("--live", action="store_true",
                         help="Also open a live preview window while saving to disk")

    args = parser.parse_args()

    def progress(current, total, phase="", fps=None, device=None):
        pct = (current / total * 100) if total else 0
        extra = f" | {device} | {fps:.1f} FPS" if fps is not None else ""
        print(f"\r{phase} {current}/{total} ({pct:.1f}%){extra}   ", end="", flush=True)

    print(f"Starting pipeline on: {args.input}\n")

    summary = run_pipeline(
        input_video=args.input,
        output_dir=args.output_dir,
        fastsam_model_name=args.model,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
        max_distance=args.max_distance, max_age=args.max_age,
        belt_angle_deg=(args.belt_angle_deg if args.belt_angle_deg == "auto" else float(args.belt_angle_deg)),
        line_ratio=args.line_ratio,
        direction=args.direction, hysteresis=args.hysteresis,
        stable_frames=args.stable_frames,
        roi_points=args.roi_points,
        min_area=args.min_area, max_area=args.max_area,
        min_area_ratio=args.min_area_ratio, max_area_ratio=args.max_area_ratio,
        cluster_threshold=args.cluster_threshold, cluster_device=args.cluster_device,
        confirm_min_size=args.confirm_min_size,
        average_embeddings=not args.no_average_embeddings,
        avg_embedding_interval=args.avg_embedding_interval,
        avg_embedding_max_samples=args.avg_embedding_max_samples,
        show_live=args.live,
        progress_callback=progress,
    )

    print("\n\n" + "=" * 40)
    print(f"Förderband erkannt als: Winkel={summary['detected_angle_deg']} Grad, Richtung={summary['detected_direction']}")
    print("-" * 40)
    print(f"Gesamt: {summary['gesamt']}")
    print()
    for name, count in summary["counts"].items():
        print(f"{name}: {count}")
    print("=" * 40)
    print(f"\nVideo: {summary['video_path']}")
    print(f"JSON:  {summary['json_path']}")
    print(f"CSV:   {summary['csv_path']}")
    print(f"Crops: {summary['clusters_dir']}/")
    print(f"ZIP:   {summary['zip_path']}")


if __name__ == "__main__":
    main()
