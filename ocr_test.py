from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import easyocr
import pandas as pd
import re
from itertools import product


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "images"
    / "test"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "plate_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "ocr_test_v3"
)

CROPS_DIR = OUTPUT_DIR / "plate_crops"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 640

YOLO_CONFIDENCE = 0.20

# We use multiple crops instead of trusting one crop.
PADDING_VALUES = [0, 2, 4]

# Only characters normally found on a registration plate.
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


# ============================================================
# COMMON INDIAN STATE / UT PREFIXES
# ============================================================

STATE_PREFIXES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL",
    "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL",
    "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB"
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Keep only A-Z and 0-9.
    """

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )


# ============================================================
# OCR CHARACTER CORRECTION MAPS
# ============================================================

LETTER_FIXES = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B"
}

DIGIT_FIXES = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8"
}


# ============================================================
# APPLY POSITION-AWARE CORRECTION
# ============================================================

def correct_segment(text, expected_type):
    """
    Correct common OCR confusions according to whether
    a segment should contain letters or digits.
    """

    corrected = []

    for char in text:

        if expected_type == "letter":

            corrected.append(
                LETTER_FIXES.get(
                    char,
                    char
                )
            )

        elif expected_type == "digit":

            corrected.append(
                DIGIT_FIXES.get(
                    char,
                    char
                )
            )

        else:

            corrected.append(char)

    return "".join(corrected)


# ============================================================
# GENERATE PLATE STRUCTURES
# ============================================================

def generate_structures(candidate):
    """
    Try plausible Indian registration-number structures.

    General form:
        SS DD AAA DDDD

    where:
        SS   = state/UT prefix
        DD   = district digits
        AAA  = series letters
        DDDD = registration digits

    The actual number of digits/letters can vary, so several
    layouts are tested.
    """

    candidate = normalize_text(candidate)

    if len(candidate) < 7:
        return []

    if len(candidate) > 12:
        candidate = candidate[:12]

    if len(candidate) < 2:
        return []

    prefix = candidate[:2]

    # Prefix should be a known state/UT code.
    if prefix not in STATE_PREFIXES:
        return []

    remainder = candidate[2:]

    structures = []

    # --------------------------------------------------------
    # district digits: 1-2
    # series letters: 1-3
    # final digits: 1-4
    # --------------------------------------------------------

    for district_len in [1, 2]:

        for series_len in [1, 2, 3]:

            final_len = (
                len(remainder)
                - district_len
                - series_len
            )

            if not 1 <= final_len <= 4:
                continue

            district = remainder[
                :district_len
            ]

            series = remainder[
                district_len:
                district_len + series_len
            ]

            final_number = remainder[
                district_len + series_len:
            ]

            # Correct each segment according to expected type.
            district_corrected = correct_segment(
                district,
                "digit"
            )

            series_corrected = correct_segment(
                series,
                "letter"
            )

            final_corrected = correct_segment(
                final_number,
                "digit"
            )

            structures.append(
                prefix
                + district_corrected
                + series_corrected
                + final_corrected
            )

    return structures


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidate(
    raw_text,
    ocr_confidence
):
    """
    Score an OCR candidate without using ground truth.
    """

    text = normalize_text(
        raw_text
    )

    if not text:
        return 0.0, ""

    score = 0.0


    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    if 7 <= len(text) <= 12:
        score += 2.0
    else:
        score -= 2.0


    # --------------------------------------------------------
    # Known state prefix
    # --------------------------------------------------------

    if (
        len(text) >= 2
        and text[:2] in STATE_PREFIXES
    ):
        score += 5.0


    # --------------------------------------------------------
    # Generate corrected structures
    # --------------------------------------------------------

    structures = generate_structures(
        text
    )


    if structures:

        score += 5.0

        # Prefer the first generated candidate.
        best_structure = structures[0]

    else:

        best_structure = text


    # --------------------------------------------------------
    # OCR confidence
    # --------------------------------------------------------

    score += (
        max(
            0.0,
            min(
                1.0,
                ocr_confidence
            )
        )
        * 4.0
    )


    # --------------------------------------------------------
    # Penalize suspicious text
    # --------------------------------------------------------

    if len(set(text)) == 1:
        score -= 3.0

    return score, best_structure


# ============================================================
# CREATE IMAGE VARIANTS
# ============================================================

def create_variants(crop):

    variants = []


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    enlarged = cv2.resize(
        crop,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(
        (
            "original",
            enlarged
        )
    )


    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    variants.append(
        (
            "gray",
            gray
        )
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
        (
            "clahe",
            enhanced
        )
    )


    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )[1]

    variants.append(
        (
            "otsu",
            otsu
        )
    )


    # --------------------------------------------------------
    # INVERTED OTSU
    # --------------------------------------------------------

    otsu_inverted = cv2.bitwise_not(
        otsu
    )

    variants.append(
        (
            "otsu_inverted",
            otsu_inverted
        )
    )


    # --------------------------------------------------------
    # ADAPTIVE THRESHOLD
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    variants.append(
        (
            "adaptive",
            adaptive
        )
    )


    # --------------------------------------------------------
    # SHARPEN
    # --------------------------------------------------------

    sharpen_kernel = (
        __import__("numpy")
        .array(
            [
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ],
            dtype="float32"
        )
    )

    sharpened = cv2.filter2D(
        gray,
        -1,
        sharpen_kernel
    )

    variants.append(
        (
            "sharpened",
            sharpened
        )
    )


    return variants


# ============================================================
# OCR
# ============================================================

def extract_ocr_candidates(
    reader,
    image
):
    """
    Read text fragments and combine them from left to right.
    """

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist=ALLOWLIST,
        decoder="beamsearch",
        mag_ratio=1.5,
        text_threshold=0.35,
        low_text=0.2,
        link_threshold=0.35,
        width_ths=0.7,
        height_ths=0.7
    )


    if not results:
        return []


    fragments = []


    for result in results:

        if len(result) < 3:
            continue

        bbox = result[0]

        text = normalize_text(
            result[1]
        )

        confidence = float(
            result[2]
        )

        if not text:
            continue


        left_x = min(
            point[0]
            for point in bbox
        )


        fragments.append(
            (
                left_x,
                text,
                confidence
            )
        )


    if not fragments:
        return []


    fragments.sort(
        key=lambda item: item[0]
    )


    # --------------------------------------------------------
    # Individual fragments
    # --------------------------------------------------------

    candidates = []

    for _, text, confidence in fragments:

        candidates.append(
            (
                text,
                confidence
            )
        )


    # --------------------------------------------------------
    # Combined text
    # --------------------------------------------------------

    combined = "".join(
        fragment[1]
        for fragment in fragments
    )

    average_confidence = (
        sum(
            fragment[2]
            for fragment in fragments
        )
        / len(fragments)
    )


    candidates.append(
        (
            combined,
            average_confidence
        )
    )


    return candidates


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_distance(a, b):

    rows = len(a) + 1
    cols = len(b) + 1

    dp = [
        [0] * cols
        for _ in range(rows)
    ]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):

        for j in range(1, cols):

            cost = (
                0
                if a[i - 1] == b[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


# ============================================================
# CHARACTER SIMILARITY
# ============================================================

def character_similarity(
    predicted,
    actual
):

    if not actual:
        return 0.0

    distance = edit_distance(
        predicted,
        actual
    )

    return max(
        0.0,
        1.0
        - (
            distance
            / max(
                len(predicted),
                len(actual)
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("NUMBER PLATE OCR - VERSION 3")
    print("========================================")


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not TEST_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Test directory not found:\n"
            f"{TEST_IMAGE_DIR}"
        )


    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CROPS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    metadata = None

    if METADATA_PATH.exists():

        metadata = pd.read_csv(
            METADATA_PATH
        )

        metadata["image"] = (
            metadata["image"]
            .astype(str)
            .str.lower()
        )

        print(
            "\nGround-truth metadata:",
            len(metadata)
        )


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = 0
        use_gpu = True

        print("\nCUDA: YES")
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = "cpu"
        use_gpu = False

        print("\nCUDA: NO")
        print("Using CPU.")


    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    print("\nLoading YOLO...")

    model = YOLO(
        str(MODEL_PATH)
    )


    # --------------------------------------------------------
    # LOAD OCR
    # --------------------------------------------------------

    print("\nLoading EasyOCR...")

    reader = easyocr.Reader(
        ["en"],
        gpu=use_gpu
    )


    # --------------------------------------------------------
    # TEST IMAGES
    # --------------------------------------------------------

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    test_images = sorted(
        [
            path
            for path in TEST_IMAGE_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in valid_extensions
        ]
    )


    print(
        "\nTest images:",
        len(test_images)
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    rows = []

    exact_matches = 0

    evaluated_images = 0

    total_similarity = 0.0


    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for index, image_path in enumerate(
        test_images,
        start=1
    ):

        print(
            f"\n[{index}/{len(test_images)}]"
            f" {image_path.name}"
        )


        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue


        height, width = image.shape[:2]


        # ----------------------------------------------------
        # YOLO DETECTION
        # ----------------------------------------------------

        results = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            device=device,
            verbose=False
        )


        if not results:
            continue


        result = results[0]


        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            print("No plate detected.")

            continue


        # ----------------------------------------------------
        # SELECT BEST PLATE
        # ----------------------------------------------------

        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        best_index = confidences.argmax()

        yolo_confidence = float(
            confidences[best_index]
        )


        box = (
            result.boxes.xyxy[
                best_index
            ]
            .detach()
            .cpu()
            .numpy()
        )


        x1, y1, x2, y2 = [
            int(v)
            for v in box
        ]


        # ----------------------------------------------------
        # OCR CANDIDATES
        # ----------------------------------------------------

        all_candidates = []


        for padding in PADDING_VALUES:

            px1 = max(
                0,
                x1 - padding
            )

            py1 = max(
                0,
                y1 - padding
            )

            px2 = min(
                width,
                x2 + padding
            )

            py2 = min(
                height,
                y2 + padding
            )


            crop = image[
                py1:py2,
                px1:px2
            ]


            if crop.size == 0:
                continue


            # Save tight crop
            if padding == 0:

                crop_path = (
                    CROPS_DIR
                    / f"{image_path.stem}_plate.jpg"
                )

                cv2.imwrite(
                    str(crop_path),
                    crop
                )


            variants = create_variants(
                crop
            )


            for variant_name, variant_image in variants:

                ocr_candidates = (
                    extract_ocr_candidates(
                        reader,
                        variant_image
                    )
                )


                for (
                    raw_text,
                    confidence
                ) in ocr_candidates:

                    score, corrected = (
                        score_candidate(
                            raw_text,
                            confidence
                        )
                    )


                    all_candidates.append({
                        "raw_text":
                            normalize_text(
                                raw_text
                            ),

                        "corrected":
                            corrected,

                        "confidence":
                            confidence,

                        "score":
                            score,

                        "padding":
                            padding,

                        "variant":
                            variant_name
                    })


        # ----------------------------------------------------
        # SELECT BEST CANDIDATE
        # ----------------------------------------------------

        if not all_candidates:

            best_text = ""

            best_ocr_confidence = 0.0

            best_variant = "none"

            best_score = 0.0

        else:

            all_candidates.sort(
                key=lambda item: (
                    item["score"],
                    item["confidence"]
                ),
                reverse=True
            )


            best = all_candidates[0]


            best_text = best["corrected"]

            best_ocr_confidence = (
                best["confidence"]
            )

            best_variant = (
                f'{best["variant"]}'
                f'_pad{best["padding"]}'
            )

            best_score = (
                best["score"]
            )


        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        ground_truth = ""


        if metadata is not None:

            match = metadata[
                metadata["image"]
                == image_path.name.lower()
            ]


            if not match.empty:

                ground_truth = normalize_text(
                    str(
                        match.iloc[0][
                            "plate_text"
                        ]
                    )
                )


        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        exact_match = False

        similarity = 0.0


        if ground_truth:

            evaluated_images += 1


            exact_match = (
                best_text == ground_truth
            )


            if exact_match:

                exact_matches += 1


            similarity = (
                character_similarity(
                    best_text,
                    ground_truth
                )
            )


            total_similarity += similarity


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            "YOLO confidence:",
            f"{yolo_confidence:.3f}"
        )

        print(
            "OCR result:",
            best_text
            if best_text
            else "NO TEXT"
        )

        print(
            "OCR confidence:",
            f"{best_ocr_confidence:.3f}"
        )

        print(
            "Selected variant:",
            best_variant
        )


        if ground_truth:

            print(
                "Ground truth:",
                ground_truth
            )

            print(
                "Character similarity:",
                f"{similarity * 100:.1f}%"
            )

            print(
                "Exact match:",
                exact_match
            )


        # ----------------------------------------------------
        # SHOW TOP CANDIDATES
        # ----------------------------------------------------

        print(
            "Top OCR candidates:"
        )


        for candidate in all_candidates[:3]:

            print(
                " ",
                candidate["raw_text"],
                "->",
                candidate["corrected"],
                f"(score={candidate['score']:.2f},",
                f"conf={candidate['confidence']:.2f},",
                f"{candidate['variant']},",
                f"pad={candidate['padding']})"
            )


        # ----------------------------------------------------
        # ANNOTATED IMAGE
        # ----------------------------------------------------

        annotated = image.copy()


        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        display_text = (
            best_text
            if best_text
            else "NO TEXT"
        )


        cv2.putText(
            annotated,
            display_text,
            (
                x1,
                max(
                    30,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        output_path = (
            OUTPUT_DIR
            / image_path.name
        )


        cv2.imwrite(
            str(output_path),
            annotated
        )


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        rows.append({

            "image":
                image_path.name,

            "yolo_confidence":
                round(
                    yolo_confidence,
                    4
                ),

            "ocr_text":
                best_text,

            "ocr_confidence":
                round(
                    best_ocr_confidence,
                    4
                ),

            "variant":
                best_variant,

            "ground_truth":
                ground_truth,

            "exact_match":
                exact_match,

            "character_similarity":
                round(
                    similarity,
                    4
                )
        })


    # ========================================================
    # SAVE RESULTS CSV
    # ========================================================

    dataframe = pd.DataFrame(
        rows
    )


    csv_path = (
        OUTPUT_DIR
        / "ocr_results.csv"
    )


    dataframe.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n========================================")
    print("OCR V3 COMPLETE")
    print("========================================")


    print(
        "Images processed:",
        len(rows)
    )


    print(
        "Results CSV:",
        csv_path
    )


    if evaluated_images > 0:

        exact_accuracy = (
            exact_matches
            / evaluated_images
        )

        average_similarity = (
            total_similarity
            / evaluated_images
        )


        print(
            "\nExact-match accuracy:",
            f"{exact_accuracy * 100:.2f}%"
        )

        print(
            "Character similarity:",
            f"{average_similarity * 100:.2f}%"
        )

        print(
            "Exact matches:",
            exact_matches,
            "/",
            evaluated_images
        )