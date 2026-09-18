import random
import shutil
import csv
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20

SOURCE_ROOT = Path("NumberPlate_Annotated_Dataset")
IMAGE_DIR = SOURCE_ROOT / "Images"
LABEL_DIR = SOURCE_ROOT / "Labels"

OUTPUT_ROOT = Path("dataset")

CLASS_ID = 0
CLASS_NAME = "number_plate"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

random.seed(SEED)


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(stem):

    # Check directly inside Images/
    for ext in IMAGE_EXTENSIONS:

        path = IMAGE_DIR / f"{stem}{ext}"

        if path.exists():
            return path

    # Search inside subfolders
    for path in IMAGE_DIR.rglob("*"):

        if path.is_file() and path.stem == stem:

            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return path

    return None


# ============================================================
# READ XML
# ============================================================

def read_xml(xml_path):

    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename")

    objects = []

    for obj in root.findall("object"):

        plate_text = obj.findtext(
            "name",
            default="UNKNOWN"
        )

        xmin = float(
            obj.findtext("bndbox/xmin")
        )

        ymin = float(
            obj.findtext("bndbox/ymin")
        )

        xmax = float(
            obj.findtext("bndbox/xmax")
        )

        ymax = float(
            obj.findtext("bndbox/ymax")
        )

        objects.append({
            "plate_text": plate_text,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

    return filename, objects


# ============================================================
# VOC → YOLO
# ============================================================

def voc_to_yolo(
    xmin,
    ymin,
    xmax,
    ymax,
    image_width,
    image_height
):

    x_center = (
        (xmin + xmax) / 2
    ) / image_width

    y_center = (
        (ymin + ymax) / 2
    ) / image_height

    box_width = (
        xmax - xmin
    ) / image_width

    box_height = (
        ymax - ymin
    ) / image_height

    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, box_width)),
        max(0.0, min(1.0, box_height))
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:

    (
        OUTPUT_ROOT
        / "images"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_ROOT
        / "labels"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FIND XML FILES
# ============================================================

xml_files = sorted(
    LABEL_DIR.rglob("*.xml")
)

if not xml_files:

    raise FileNotFoundError(
        f"No XML files found in {LABEL_DIR}"
    )

print(
    "XML annotations found:",
    len(xml_files)
)


# ============================================================
# FILTER XML FILES WITH MISSING IMAGES
# ============================================================

valid_pairs = []
missing_images = []

for xml_path in xml_files:

    try:

        filename, objects = read_xml(xml_path)

        image_path = find_image(
            Path(filename).stem
        )

        if image_path is None:

            missing_images.append(
                f"{xml_path.name}: image not found"
            )

            continue

        valid_pairs.append(
            (xml_path, image_path, filename, objects)
        )

    except Exception as e:

        missing_images.append(
            f"{xml_path.name}: {e}"
        )


print(
    "\nValid image/XML pairs:",
    len(valid_pairs)
)

print(
    "Missing/invalid:",
    len(missing_images)
)


# ============================================================
# SPLIT DATASET
# ============================================================

random.shuffle(valid_pairs)

total = len(valid_pairs)

train_end = int(
    total * TRAIN_RATIO
)

val_end = train_end + int(
    total * VAL_RATIO
)

splits = {

    "train": valid_pairs[:train_end],

    "val": valid_pairs[train_end:val_end],

    "test": valid_pairs[val_end:]
}


print("\nDataset split:")

for split, files in splits.items():

    print(
        f"{split}: {len(files)}"
    )


# ============================================================
# CONVERT XML → YOLO
# ============================================================

metadata = []
skipped = []

for split, files in splits.items():

    for xml_path, image_path, filename, objects in files:

        try:

            # ------------------------------------------------
            # READ ACTUAL IMAGE
            # ------------------------------------------------

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                skipped.append(
                    f"{xml_path.name}: unable to read image"
                )

                continue

            # OpenCV returns:
            # image.shape = height, width, channels

            image_height, image_width = image.shape[:2]


            # ------------------------------------------------
            # PROCESS OBJECTS
            # ------------------------------------------------

            yolo_lines = []

            for obj in objects:

                xmin = obj["xmin"]
                ymin = obj["ymin"]
                xmax = obj["xmax"]
                ymax = obj["ymax"]


                # --------------------------------------------
                # CHECK BOUNDING BOX
                # --------------------------------------------

                if (
                    xmax <= xmin
                    or
                    ymax <= ymin
                ):

                    skipped.append(
                        f"{xml_path.name}: invalid bounding box"
                    )

                    continue


                # --------------------------------------------
                # CHECK BOX AGAINST ACTUAL IMAGE SIZE
                # --------------------------------------------

                if (
                    xmin < 0
                    or
                    ymin < 0
                    or
                    xmax > image_width
                    or
                    ymax > image_height
                ):

                    print(
                        f"\nWARNING: {xml_path.name}"
                    )

                    print(
                        "Bounding box exceeds image dimensions."
                    )

                    print(
                        f"Image size: "
                        f"{image_width} x {image_height}"
                    )

                    print(
                        f"Box: "
                        f"{xmin}, {ymin}, "
                        f"{xmax}, {ymax}"
                    )

                    # Clip box to image boundaries

                    xmin = max(
                        0,
                        min(xmin, image_width)
                    )

                    ymin = max(
                        0,
                        min(ymin, image_height)
                    )

                    xmax = max(
                        0,
                        min(xmax, image_width)
                    )

                    ymax = max(
                        0,
                        min(ymax, image_height)
                    )


                # --------------------------------------------
                # CONVERT VOC → YOLO
                # --------------------------------------------

                (
                    x_center,
                    y_center,
                    box_width,
                    box_height
                ) = voc_to_yolo(
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    image_width,
                    image_height
                )


                # --------------------------------------------
                # YOLO FORMAT
                #
                # class
                # x_center
                # y_center
                # width
                # height
                # --------------------------------------------

                yolo_lines.append(
                    f"{CLASS_ID} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}"
                )


                # --------------------------------------------
                # SAVE METADATA
                # --------------------------------------------

                metadata.append({

                    "split": split,

                    "image": Path(
                        filename
                    ).name,

                    "plate_text": obj[
                        "plate_text"
                    ],

                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,

                    "width": image_width,
                    "height": image_height
                })


            # ------------------------------------------------
            # NO VALID OBJECT
            # ------------------------------------------------

            if not yolo_lines:

                skipped.append(
                    f"{xml_path.name}: no valid object"
                )

                continue


            # ------------------------------------------------
            # COPY IMAGE
            # ------------------------------------------------

            output_image = (
                OUTPUT_ROOT
                / "images"
                / split
                / Path(filename).name
            )

            shutil.copy2(
                image_path,
                output_image
            )


            # ------------------------------------------------
            # WRITE YOLO LABEL
            # ------------------------------------------------

            output_label = (
                OUTPUT_ROOT
                / "labels"
                / split
                / f"{Path(filename).stem}.txt"
            )

            output_label.write_text(
                "\n".join(yolo_lines) + "\n",
                encoding="utf-8"
            )


        except Exception as e:

            skipped.append(
                f"{xml_path.name}: {e}"
            )


# ============================================================
# SAVE PLATE METADATA
# ============================================================

metadata_path = (
    OUTPUT_ROOT
    / "plate_metadata.csv"
)

fieldnames = [
    "split",
    "image",
    "plate_text",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "width",
    "height"
]

with metadata_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        metadata
    )


# ============================================================
# CREATE YOLO data.yaml
# ============================================================

yaml_content = (
    "path: .\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n\n"
    "names:\n"
    "  0: number_plate\n"
)

(
    OUTPUT_ROOT
    / "data.yaml"
).write_text(
    yaml_content,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n========== DATASET PREPARATION COMPLETE =========="
)

print(
    "Converted annotations:",
    len(metadata)
)

print(
    "Skipped items:",
    len(skipped)
)

print(
    "\nClass:"
)

print(
    "0 = number_plate"
)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_ROOT.resolve()
)


# ============================================================
# SHOW SKIPPED ITEMS
# ============================================================

all_skipped = (
    missing_images
    + skipped
)

if all_skipped:

    print(
        "\nSkipped items:"
    )

    for item in all_skipped[:20]:

        print(
            " -",
            item
        )