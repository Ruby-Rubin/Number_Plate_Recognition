from pathlib import Path
import cv2

IMAGE_DIR = Path("dataset/images/train")
LABEL_DIR = Path("dataset/labels/train")

# Pick one image
image_path = next(IMAGE_DIR.iterdir())

label_path = LABEL_DIR / f"{image_path.stem}.txt"

print("Image :", image_path)
print("Label :", label_path)

# Read image
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("Could not read image.")

height, width = image.shape[:2]

# Read YOLO label
with open(label_path, "r") as file:
    lines = file.readlines()

for line in lines:

    values = line.strip().split()

    class_id = int(values[0])
    x_center = float(values[1])
    y_center = float(values[2])
    box_width = float(values[3])
    box_height = float(values[4])

    # Convert YOLO → pixel coordinates
    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    xmin = int(x_center - box_width / 2)
    ymin = int(y_center - box_height / 2)
    xmax = int(x_center + box_width / 2)
    ymax = int(y_center + box_height / 2)

    cv2.rectangle(
        image,
        (xmin, ymin),
        (xmax, ymax),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        "number_plate",
        (xmin, max(ymin - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

cv2.imwrite(
    "dataset_verification.jpg",
    image
)

cv2.imshow(
    "YOLO Dataset Verification",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()
