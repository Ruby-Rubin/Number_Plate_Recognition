from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import torch
import easyocr
import pandas as pd
import re

# Reuse our V4 plate-format correction logic
from ocr_postprocess_v4 import (
    learn_patterns,
    correct_plate,
    normalize
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best.pt"
)

VIDEO_DIR = (
    PROJECT_ROOT
    / "videos"
)

VIDEO_PATH = (
    VIDEO_DIR
    / "traffic.mp4"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "plate_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "video_analysis"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "traffic_annotated.mp4"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "detected_plates.csv"
)

CROPS_DIR = (
    OUTPUT_DIR
    / "plate_crops"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 640

YOLO_CONFIDENCE = 0.20

OCR_INTERVAL = 5

OCR_ALLOWLIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

MIN_CROP_WIDTH = 30
MIN_CROP_HEIGHT = 8

CROP_PADDING = 2


# ============================================================
# OCR PREPROCESSING
# ============================================================

def create_ocr_variants(crop):

    variants = []

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    enlarged = cv2.resize(
        crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(
        ("original", enlarged)
    )


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    variants.append(
        ("gray", gray)
    )


    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    variants.append(
        ("clahe", enhanced)
    )


    return variants


# ============================================================
# OCR
# ============================================================

def read_plate(reader, crop):

    candidates = []

    variants = create_ocr_variants(
        crop
    )


    for variant_name, variant in variants:

        try:

            results = reader.readtext(

                variant,

                detail=1,

                paragraph=False,

                allowlist=OCR_ALLOWLIST,

                decoder="beamsearch",

                mag_ratio=1.5,

                text_threshold=0.35,

                low_text=0.20,

                link_threshold=0.35,

                width_ths=0.7,

                height_ths=0.7
            )

        except Exception:

            continue


        for result in results:

            if len(result) < 3:
                continue


            text = normalize(
                result[1]
            )

            confidence = float(
                result[2]
            )


            if not text:
                continue


            candidates.append({

                "text": text,

                "confidence": confidence,

                "variant": variant_name
            })


    if not candidates:

        return "", 0.0, "none"


    # --------------------------------------------------------
    # Apply plate-format correction
    # --------------------------------------------------------

    corrected_candidates = []


    for candidate in candidates:

        corrected_text, correction_score, pattern = (
            correct_plate(
                candidate["text"],
                PLATE_PATTERNS
            )
        )


        # Combine OCR confidence with format score.
        #
        # Higher OCR confidence = better.
        # Lower correction score = better.
        #

        score = (
            candidate["confidence"] * 5.0
            - correction_score * 0.25
        )


        corrected_candidates.append({

            "text": corrected_text,

            "raw_text": candidate["text"],

            "confidence":
                candidate["confidence"],

            "score": score,

            "variant":
                candidate["variant"],

            "pattern":
                pattern
        })


    # --------------------------------------------------------
    # Select best candidate for this frame
    # --------------------------------------------------------

    corrected_candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    best = corrected_candidates[0]


    return (
        best["text"],
        best["confidence"],
        best["variant"]
    )


# ============================================================
# BEST TRACK RESULT
# ============================================================

def get_track_consensus(track_data):

    if not track_data:
        return "", 0.0


    # --------------------------------------------------------
    # Weighted voting
    # --------------------------------------------------------

    weighted_scores = defaultdict(float)

    confidence_values = defaultdict(list)


    for item in track_data:

        text = item["text"]

        confidence = item["confidence"]


        if not text:
            continue


        # Give repeated observations more weight.
        weighted_scores[text] += (
            0.5
            +
            confidence
        )


        confidence_values[text].append(
            confidence
        )


    if not weighted_scores:

        return "", 0.0


    # Highest accumulated score
    best_text = max(
        weighted_scores,
        key=weighted_scores.get
    )


    average_confidence = sum(
        confidence_values[best_text]
    ) / len(
        confidence_values[best_text]
    )


    return (
        best_text,
        average_confidence
    )


# ============================================================
# DRAW LABEL
# ============================================================

def draw_track_label(
    frame,
    x1,
    y1,
    x2,
    y2,
    track_id,
    plate_text,
    confidence
):

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if plate_text:

        label = (
            f"ID {track_id} | "
            f"{plate_text} | "
            f"{confidence:.2f}"
        )

    else:

        label = (
            f"ID {track_id} | "
            "Reading..."
        )


    text_y = max(
        30,
        y1 - 10
    )


    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("TRAFFIC VIDEO NUMBER PLATE ANALYSIS")
    print("========================================")


    # ========================================================
    # CHECK FILES
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}"
        )


    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"\nTraffic video not found:\n"
            f"{VIDEO_PATH}"
        )


    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"\nPlate metadata not found:\n"
            f"{METADATA_PATH}"
        )


    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CROPS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # LOAD TRAINING METADATA
    # ========================================================

    metadata = pd.read_csv(
        METADATA_PATH
    )


    # Learn plate structures from TRAIN split only.
    global PLATE_PATTERNS

    PLATE_PATTERNS = learn_patterns(
        metadata
    )


    print(
        "\nLearned state patterns:",
        len(PLATE_PATTERNS)
    )


    # ========================================================
    # DEVICE
    # ========================================================

    if torch.cuda.is_available():

        device = 0

        print("\nCUDA: YES")

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = "cpu"

        print("\nCUDA: NO")
        print("Using CPU.")


    # ========================================================
    # LOAD YOLO
    # ========================================================

    print("\nLoading YOLO model...")

    model = YOLO(
        str(MODEL_PATH)
    )


    # ========================================================
    # LOAD OCR
    # ========================================================

    print("\nLoading EasyOCR...")

    reader = easyocr.Reader(
        ["en"],
        gpu=torch.cuda.is_available()
    )


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    print("\nOpening video:")

    print(
        VIDEO_PATH
    )


    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open traffic video."
        )


    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    print("\nVideo information:")

    print(
        "Resolution:",
        f"{frame_width} x {frame_height}"
    )

    print(
        "FPS:",
        fps
    )

    print(
        "Total frames:",
        total_frames
    )


    # ========================================================
    # OUTPUT VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (
            frame_width,
            frame_height
        )
    )


    if not writer.isOpened():

        raise RuntimeError(
            "Could not create output video."
        )


    # ========================================================
    # TRACK DATA
    # ========================================================

    track_observations = defaultdict(list)

    last_ocr_frame = {}

    track_crops_saved = defaultdict(int)

    frame_number = 0


    # ========================================================
    # MAIN VIDEO LOOP
    # ========================================================

    while True:

        success, frame = cap.read()


        if not success:
            break


        frame_number += 1


        # ----------------------------------------------------
        # TRACK PLATES
        # ----------------------------------------------------

        results = model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            imgsz=IMAGE_SIZE,

            conf=YOLO_CONFIDENCE,

            device=device,

            verbose=False
        )


        if not results:

            writer.write(frame)

            continue


        result = results[0]


        # ----------------------------------------------------
        # CHECK TRACK IDS
        # ----------------------------------------------------

        if (
            result.boxes is None
            or
            not result.boxes.is_track
        ):

            writer.write(frame)

            continue


        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )


        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )


        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # PROCESS EACH TRACK
        # ----------------------------------------------------

        for box, track_id, detection_confidence in zip(
            boxes,
            track_ids,
            confidences
        ):

            x1, y1, x2, y2 = [
                int(value)
                for value in box
            ]


            # ------------------------------------------------
            # Clamp coordinates
            # ------------------------------------------------

            x1 = max(
                0,
                min(
                    x1,
                    frame_width - 1
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    frame_height - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    frame_width
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    frame_height
                )
            )


            crop_width = x2 - x1
            crop_height = y2 - y1


            # ------------------------------------------------
            # Current recognized plate
            # ------------------------------------------------

            plate_text, plate_confidence = (
                get_track_consensus(
                    track_observations[
                        track_id
                    ]
                )
            )


            # ------------------------------------------------
            # OCR every N frames per track
            # ------------------------------------------------

            should_run_ocr = (

                (
                    track_id
                    not in last_ocr_frame
                )

                or

                (
                    frame_number
                    -
                    last_ocr_frame[
                        track_id
                    ]
                    >= OCR_INTERVAL
                )
            )


            if (
                should_run_ocr
                and
                crop_width >= MIN_CROP_WIDTH
                and
                crop_height >= MIN_CROP_HEIGHT
            ):

                crop = frame[
                    max(
                        0,
                        y1 - CROP_PADDING
                    ):
                    min(
                        frame_height,
                        y2 + CROP_PADDING
                    ),

                    max(
                        0,
                        x1 - CROP_PADDING
                    ):
                    min(
                        frame_width,
                        x2 + CROP_PADDING
                    )
                ]


                if crop.size > 0:

                    text, ocr_confidence, variant = (
                        read_plate(
                            reader,
                            crop
                        )
                    )


                    last_ocr_frame[
                        track_id
                    ] = frame_number


                    if text:

                        track_observations[
                            track_id
                        ].append({

                            "text":
                                text,

                            "confidence":
                                ocr_confidence,

                            "frame":
                                frame_number,

                            "detection_confidence":
                                float(
                                    detection_confidence
                                ),

                            "variant":
                                variant
                        })


                        # Limit memory for very long videos.
                        if len(
                            track_observations[
                                track_id
                            ]
                        ) > 100:

                            track_observations[
                                track_id
                            ] = (
                                track_observations[
                                    track_id
                                ][-100:]
                            )


                        # Save only a few crops per track.
                        if (
                            track_crops_saved[
                                track_id
                            ] < 5
                        ):

                            crop_filename = (
                                f"track_{track_id}_"
                                f"frame_{frame_number}.jpg"
                            )


                            crop_path = (
                                CROPS_DIR
                                / crop_filename
                            )


                            cv2.imwrite(
                                str(crop_path),
                                crop
                            )


                            track_crops_saved[
                                track_id
                            ] += 1


            # ------------------------------------------------
            # Update consensus after OCR
            # ------------------------------------------------

            plate_text, plate_confidence = (
                get_track_consensus(
                    track_observations[
                        track_id
                    ]
                )
            )


            # ------------------------------------------------
            # Draw
            # ------------------------------------------------

            draw_track_label(

                frame,

                x1,
                y1,
                x2,
                y2,

                track_id,

                plate_text,

                plate_confidence
            )


        # ----------------------------------------------------
        # Add frame number
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Frame: {frame_number}/{total_frames}",

            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2
        )


        # ----------------------------------------------------
        # Write frame
        # ----------------------------------------------------

        writer.write(
            frame
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            frame_number % 50 == 0
        ):

            progress = (
                frame_number
                /
                max(
                    1,
                    total_frames
                )
                * 100
            )


            print(
                f"Progress: "
                f"{progress:.1f}% "
                f"({frame_number}/{total_frames})"
            )


    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()

    writer.release()


    # ========================================================
    # SAVE FINAL TRACK RESULTS
    # ========================================================

    final_rows = []


    for track_id, observations in (
        track_observations.items()
    ):

        final_text, final_confidence = (
            get_track_consensus(
                observations
            )
        )


        if not final_text:
            final_text = "UNKNOWN"


        # Count observations
        observation_count = len(
            observations
        )


        final_rows.append({

            "track_id":
                track_id,

            "plate_number":
                final_text,

            "confidence":
                round(
                    final_confidence,
                    4
                ),

            "ocr_observations":
                observation_count
        })


    final_df = pd.DataFrame(
        final_rows
    )


    final_df = final_df.sort_values(
        "track_id"
    )


    final_df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n========================================")
    print("VIDEO ANALYSIS COMPLETE")
    print("========================================")


    print(
        "Output video:"
    )

    print(
        OUTPUT_VIDEO.resolve()
    )


    print(
        "\nResults CSV:"
    )

    print(
        OUTPUT_CSV.resolve()
    )


    print(
        "\nUnique tracked plates:",
        len(final_rows)
    )


    print(
        "\nDetected plate results:"
    )


    for row in final_rows:

        print(
            f'  ID {row["track_id"]}: '
            f'{row["plate_number"]} '
            f'({row["confidence"]:.2f})'
        )