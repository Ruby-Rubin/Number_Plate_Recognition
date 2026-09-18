from ultralytics import YOLO
import torch


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CHECK DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = 0
        print("GPU detected:")
        print(torch.cuda.get_device_name(0))

    else:

        device = "cpu"
        print("CUDA GPU not available.")
        print("Training will use CPU.")


    # --------------------------------------------------------
    # LOAD PRETRAINED YOLO MODEL
    # --------------------------------------------------------

    print("\nLoading YOLO model...")

    model = YOLO("yolo26n.pt")


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nStarting number plate detection training...")

    results = model.train(

        # Dataset configuration
        data="dataset/data.yaml",

        # Number of training epochs
        epochs=100,

        # Input image size
        imgsz=640,

        # Batch size
        batch=16,

        # Training device
        device=device,

        # Stop if validation performance stops improving
        patience=20,

        # Save the best model
        save=True,

        # Project folder
        project="runs",

        # Experiment name
        name="number_plate_detector",

        # Display training progress
        verbose=True
    )


    # --------------------------------------------------------
    # TRAINING COMPLETE
    # --------------------------------------------------------

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    print("\nBest model should be saved at:")

    print(
        "runs/number_plate_detector/weights/best.pt"
    )