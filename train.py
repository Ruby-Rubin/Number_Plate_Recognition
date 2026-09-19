from ultralytics import YOLO
import torch
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "yolo26n.pt"

EPOCHS = 100
IMAGE_SIZE = 640
PATIENCE = 20

PROJECT_NAME = "runs"
RUN_NAME = "number_plate_detector"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("NUMBER PLATE DETECTION - YOLO TRAINING")
    print("========================================")

    # --------------------------------------------------------
    # PROJECT PATH
    # --------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parent

    DATASET_ROOT = PROJECT_ROOT / "dataset"

    TRAIN_PATH = DATASET_ROOT / "images" / "train"
    VAL_PATH = DATASET_ROOT / "images" / "val"
    TEST_PATH = DATASET_ROOT / "images" / "test"


    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    print("\nDataset location:")
    print(DATASET_ROOT)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_ROOT}"
        )

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training images not found:\n{TRAIN_PATH}"
        )

    if not VAL_PATH.exists():
        raise FileNotFoundError(
            f"Validation images not found:\n{VAL_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test images not found:\n{TEST_PATH}"
        )

    print("\nDataset folders found successfully.")


    # --------------------------------------------------------
    # DEVICE DETECTION
    # --------------------------------------------------------

    print("\n========================================")
    print("DEVICE INFORMATION")
    print("========================================")

    if torch.cuda.is_available():

        device = 0
        batch_size = 16

        gpu_name = torch.cuda.get_device_name(0)

        print("CUDA available: YES")
        print("GPU:", gpu_name)
        print("Device: CUDA:0")

        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    else:

        device = "cpu"
        batch_size = 4

        print("CUDA available: NO")
        print("Using CPU.")
        print("PyTorch:", torch.__version__)


    # --------------------------------------------------------
    # CREATE YOLO DATA CONFIGURATION
    # --------------------------------------------------------

    data_yaml = DATASET_ROOT / "data_runtime.yaml"

    yaml_content = f"""path: {DATASET_ROOT.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: number_plate
"""

    data_yaml.write_text(
        yaml_content,
        encoding="utf-8"
    )

    print("\nYOLO dataset configuration:")
    print(data_yaml)


    # --------------------------------------------------------
    # LOAD PRETRAINED MODEL
    # --------------------------------------------------------

    print("\n========================================")
    print("LOADING YOLO MODEL")
    print("========================================")

    print("Model:", MODEL_NAME)

    model = YOLO(MODEL_NAME)


    # --------------------------------------------------------
    # START TRAINING
    # --------------------------------------------------------

    print("\n========================================")
    print("STARTING TRAINING")
    print("========================================")

    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", batch_size)
    print("Device:", device)


    results = model.train(

        data=str(data_yaml),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=batch_size,

        device=device,

        patience=PATIENCE,

        save=True,

        project=PROJECT_NAME,

        name=RUN_NAME,

        amp=True,

        verbose=True
    )


    # --------------------------------------------------------
    # TRAINING COMPLETE
    # --------------------------------------------------------

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    print("\nBest model:")
    print(
        PROJECT_ROOT
        / "runs"
        / RUN_NAME
        / "weights"
        / "best.pt"
    )


    # --------------------------------------------------------
    # CLEAN GPU MEMORY
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.empty_cache()