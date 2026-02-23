#!/usr/bin/env python3
"""
People Counter with Preview + Logging (Axis MJPEG / HLS / RTSP friendly)

- Live preview window with boxes, counting line, IN/OUT overlay
- Console logging per frame
- Optional CSV logging and saving annotated video

Dependencies:
    pip install ultralytics supervision opencv-python

Run (Axis MJPEG at UT Austin):
    python people_counter_preview.py \
      --source "http://porchcam.ece.utexas.edu/axis-cgi/mjpg/video.cgi?resolution=320x240"

Other examples:
    python people_counter_preview.py --source 0                                # webcam
    python people_counter_preview.py --source "video.mp4"                      # file
    python people_counter_preview.py --source "https://.../playlist.m3u8"      # HLS
    python people_counter_preview.py --line "960,120,960,900"                  # custom line
    python people_counter_preview.py --csv people_log.csv --save annotated.mp4 # CSV + video
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


# ----------------------------
# Defaults (edit if you like)
# ----------------------------
DEFAULT_SOURCE = "http://porchcam.ece.utexas.edu/axis-cgi/mjpg/video.cgi?resolution=320x240"
DEFAULT_MODEL = "yolov8n.pt"   # try yolov8s.pt for higher accuracy (slower)
CONF_THRES = 0.4
IOU_THRES = 0.5
PERSON_CLASS_ID = 0            # COCO class: person
DEFAULT_CSV = ""               # e.g. "people_counter_log.csv" to enable
DEFAULT_SAVE = ""              # e.g. "annotated.mp4" to enable saving
SHOW_FPS_OVERLAY = True


def parse_args():
    p = argparse.ArgumentParser(description="People counter with preview + logging")
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                   help='Source: "0" for webcam, file path, MJPEG/HLS/RTSP URL')
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO model path (e.g., yolov8n.pt)")
    p.add_argument("--conf", type=float, default=CONF_THRES, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=IOU_THRES, help="IoU threshold for NMS")
    p.add_argument("--line", type=str, default="",
                   help='Counting line as "x1,y1,x2,y2" pixels. Empty = auto vertical mid-line.')
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="CSV log path (empty = disabled)")
    p.add_argument("--save", type=str, default=DEFAULT_SAVE, help="Save annotated video to this path (empty = disabled)")
    p.add_argument("--width", type=int, default=0, help="Force output width (0=auto)")
    p.add_argument("--height", type=int, default=0, help="Force output height (0=auto)")
    return p.parse_args()


def parse_line_arg(line_arg):
    if not line_arg:
        return None, None
    try:
        x1, y1, x2, y2 = map(int, line_arg.split(","))
        return (x1, y1), (x2, y2)
    except Exception:
        print('Invalid --line format. Expected "x1,y1,x2,y2". Using auto line.')
        return None, None


def open_capture(src: str) -> cv2.VideoCapture:
    """
    Tries to open with default backend, falls back to FFmpeg (helpful for MJPEG/HLS).
    """
    # Map "0" to webcam integer 0
    cap = cv2.VideoCapture(0) if src == "0" else cv2.VideoCapture(src)
    if cap.isOpened():
        return cap

    # Fallback: explicitly request FFmpeg backend
    print("Default backend failed; retrying with CAP_FFMPEG …")
    backend = cv2.CAP_FFMPEG
    cap = cv2.VideoCapture(0, backend) if src == "0" else cv2.VideoCapture(src, backend)
    if cap.isOpened():
        return cap

    raise RuntimeError(f"Could not open source: {src}")


def main():
    args = parse_args()

    # Open source
    cap = open_capture(args.source)

    # Read props (provide sensible fallbacks for network streams)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 25.0  # a safe default for preview/record

    if args.width > 0 and args.height > 0:
        width, height = args.width, args.height

    # Prepare video writer (optional)
    writer = None
    if args.save:
        out_path = Path(args.save)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            print("⚠️ Could not open video writer; disabling save.")
            writer = None

    # Load model + tracker
    model = YOLO(args.model)
    tracker = sv.ByteTrack()

    # Counting line
    lstart, lend = parse_line_arg(args.line)
    if lstart is None or lend is None:
        # auto vertical mid-line
        x = width // 2
        lstart = (x, int(height * 0.15))
        lend = (x, int(height * 0.85))

    line_zone = sv.LineZone(start=sv.Point(*lstart), end=sv.Point(*lend))
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.6)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.8)

    # CSV logger
    csv_writer = None
    fcsv = None
    if args.csv:
        fcsv = open(args.csv, "w", newline="")
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(["ts_iso_utc", "frame", "persons_detected", "in_count", "out_count", "source"])

    print("Preview window ready. Press 'q' to quit.")
    frame_num = 0
    last_fps_t = time.time()
    fps_counter = 0
    smoothed_fps = fps

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream ended or read error.")
            break

        # Resize frame to requested output size (if forced)
        if (frame.shape[1] != width) or (frame.shape[0] != height):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        frame_num += 1
        fps_counter += 1

        # YOLO inference (person only)
        res = model.predict(frame, conf=args.conf, iou=args.iou, classes=[PERSON_CLASS_ID], verbose=False)
        dets = sv.Detections.from_ultralytics(res[0])

        # Tracking + line crossing
        dets = tracker.update_with_detections(dets)
        line_zone.trigger(dets)

        # Annotations
        labels = [f"person {c:.2f}" for c in (dets.confidence or [])]
        annotated = box_annotator.annotate(scene=frame.copy(), detections=dets, labels=labels)
        line_annotator.annotate(frame=annotated, line_counter=line_zone)

        # FPS overlay (smoothed)
        if SHOW_FPS_OVERLAY:
            now = time.time()
            if now - last_fps_t >= 0.5:
                inst = fps_counter / (now - last_fps_t)
                smoothed_fps = 0.85 * smoothed_fps + 0.15 * inst
                last_fps_t = now
                fps_counter = 0
            overlay = f"IN: {line_zone.in_count}  OUT: {line_zone.out_count}  |  FPS: {smoothed_fps:.1f}"
            # draw with outline for readability
            cv2.putText(annotated, overlay, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, overlay, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Show preview
        cv2.imshow("People Counter — Preview", annotated)

        # Console log
        persons = len(dets)
        print(f"[{frame_num}] persons={persons} | IN={line_zone.in_count} | OUT={line_zone.out_count}")

        # CSV log
        if csv_writer:
            csv_writer.writerow([
                datetime.utcnow().isoformat(),
                frame_num,
                persons,
                line_zone.in_count,
                line_zone.out_count,
                args.source,
            ])

        # Save annotated video
        if writer is not None:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Annotated video saved to: {args.save}")
    cv2.destroyAllWindows()
    if fcsv:
        fcsv.close()
        print(f"CSV saved: {args.csv}")

    print("Done.")


if __name__ == "__main__":
    main()
