from ultralytics import YOLO
from pathlib import Path
import torch


# ============================================================
# SETTINGS
# ============================================================

# Exact best model produced by your training run
BEST_MODEL = (
    r"C:\Users\Geetha P\OneDrive\Desktop\Ruby-Projects"
    r"\Brand-Logo-Recognizer\runs\detect\runs"
    r"\number_plate_detector\weights\best.pt"
)

DATASET_CONFIG = "dataset/data_runtime.yaml"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("NUMBER PLATE DETECTOR - TEST EVALUATION")
    print("========================================")


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    model_path = Path(BEST_MODEL)

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nBest model not found:\n{model_path}"
        )

    print("\nModel:")
    print(model_path)


    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    data_path = Path(DATASET_CONFIG)

    if not data_path.exists():
        raise FileNotFoundError(
            f"\nDataset configuration not found:\n{data_path}"
        )

    print("\nDataset:")
    print(data_path)


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

    model = YOLO(str(model_path))


    # --------------------------------------------------------
    # TEST SET EVALUATION
    # --------------------------------------------------------

    print("\n========================================")
    print("EVALUATING ON TEST SET")
    print("========================================")

    metrics = model.val(

        data=str(data_path),

        split="test",

        imgsz=640,

        device=device,

        plots=True,

        verbose=True
    )


    # --------------------------------------------------------
    # EXTRACT METRICS
    # --------------------------------------------------------

    precision = metrics.box.p[0]
    recall = metrics.box.r[0]
    map50 = metrics.box.map50
    map50_95 = metrics.box.map


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL TEST RESULTS")
    print("========================================")

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"mAP@50      : {map50:.4f}"
    )

    print(
        f"mAP@50-95   : {map50_95:.4f}"
    )


    # --------------------------------------------------------
    # PERCENTAGE FORMAT
    # --------------------------------------------------------

    print("\n========================================")
    print("PERCENTAGE")
    print("========================================")

    print(
        f"Precision   : {precision * 100:.2f}%"
    )

    print(
        f"Recall      : {recall * 100:.2f}%"
    )

    print(
        f"mAP@50      : {map50 * 100:.2f}%"
    )

    print(
        f"mAP@50-95   : {map50_95 * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVED RESULTS
    # --------------------------------------------------------

    print("\n========================================")
    print("TEST EVALUATION COMPLETE")
    print("========================================")

    print(
        "\nUltralytics has also generated evaluation"
        "\nplots such as the confusion matrix and"
        "\nprecision-recall curves."
    )