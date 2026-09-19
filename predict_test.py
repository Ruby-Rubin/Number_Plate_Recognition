from ultralytics import YOLO
from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "dataset"
TEST_IMAGE_DIR = DATASET_ROOT / "images" / "test"

MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_DIR_NAME = "test_predictions"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 640
CONFIDENCE = 0.25


# ============================================================
# FIND BEST MODEL
# ============================================================

def find_best_model():

    possible_models = [
        MODELS_DIR / "best.pt",
        WEIGHTS_DIR / "best.pt",
    ]

    for model_path in possible_models:

        if model_path.exists():
            return model_path

    return None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("NUMBER PLATE DETECTION - IMAGE TEST")
    print("========================================")


    # --------------------------------------------------------
    # FIND MODEL
    # --------------------------------------------------------

    MODEL_PATH = find_best_model()

    if MODEL_PATH is None:

        print("\nERROR: best.pt was not found.")

        print("\nChecked locations:")

        print(
            " -",
            MODELS_DIR / "best.pt"
        )

        print(
            " -",
            WEIGHTS_DIR / "best.pt"
        )

        raise FileNotFoundError(
            "\nPlease place your trained best.pt inside "
            "'models' or 'weights'."
        )


    print("\nModel:")
    print(
        MODEL_PATH.resolve()
    )


    # --------------------------------------------------------
    # CHECK TEST IMAGE DIRECTORY
    # --------------------------------------------------------

    if not TEST_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"\nTest image folder not found:\n"
            f"{TEST_IMAGE_DIR}"
        )


    # --------------------------------------------------------
    # COUNT TEST IMAGES
    # --------------------------------------------------------

    valid_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    ]

    test_images = [
        path
        for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    ]


    print("\nTest image folder:")
    print(
        TEST_IMAGE_DIR.resolve()
    )

    print(
        "\nTest images:",
        len(test_images)
    )


    if len(test_images) == 0:

        raise RuntimeError(
            "\nNo test images were found."
        )


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = 0

        print("\nCUDA available: YES")

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = "cpu"

        print("\nCUDA available: NO")
        print("Using CPU.")


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\n========================================")
    print("LOADING TRAINED MODEL")
    print("========================================")

    model = YOLO(
        str(MODEL_PATH)
    )


    # --------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------

    print("\n========================================")
    print("RUNNING NUMBER PLATE DETECTION")
    print("========================================")

    results = model.predict(

        source=str(TEST_IMAGE_DIR),

        imgsz=IMAGE_SIZE,

        conf=CONFIDENCE,

        device=device,

        save=True,

        project=str(RUNS_DIR),

        name=OUTPUT_DIR_NAME,

        exist_ok=True,

        verbose=True
    )


    # --------------------------------------------------------
    # COUNT DETECTIONS
    # --------------------------------------------------------

    total_detections = 0

    images_with_detection = 0

    images_without_detection = 0


    for result in results:

        if result.boxes is not None:

            detection_count = len(
                result.boxes
            )

        else:

            detection_count = 0


        total_detections += detection_count


        if detection_count > 0:

            images_with_detection += 1

        else:

            images_without_detection += 1


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_directory = (
        RUNS_DIR / OUTPUT_DIR_NAME
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n========================================")
    print("PREDICTION COMPLETE")
    print("========================================")

    print(
        "Images processed:",
        len(results)
    )

    print(
        "Images with detection:",
        images_with_detection
    )

    print(
        "Images without detection:",
        images_without_detection
    )

    print(
        "Total plates detected:",
        total_detections
    )

    print("\nAnnotated images saved to:")

    print(
        output_directory.resolve()
    )

    print("\n========================================")
    print("NEXT STEP")
    print("========================================")

    print(
        "Open the test_predictions folder and "
        "visually inspect the detected boxes."
    )   