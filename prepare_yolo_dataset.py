import os
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image

# =========================================================
# PATHS
# =========================================================

GTSRB_PATH = Path(r"C:\Users\USER\Downloads\archive")

TRAIN_CSV = GTSRB_PATH / "Train.csv"
TRAIN_FOLDER = GTSRB_PATH / "Train"

OUTPUT_PATH = Path("yolo_dataset")

# =========================================================
# CREATE FOLDERS
# =========================================================

images_train = OUTPUT_PATH / "images" / "train"
images_val = OUTPUT_PATH / "images" / "val"

labels_train = OUTPUT_PATH / "labels" / "train"
labels_val = OUTPUT_PATH / "labels" / "val"

for folder in [
    images_train,
    images_val,
    labels_train,
    labels_val
]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD CSV
# =========================================================

print("Loading GTSRB Train.csv...")

df = pd.read_csv(TRAIN_CSV)

print(f"Total images found: {len(df)}")

# =========================================================
# SHUFFLE DATA
# =========================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# 80% training / 20% validation
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
val_df = df.iloc[split_index:]

print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")

# =========================================================
# FUNCTION TO CONVERT ONE IMAGE
# =========================================================

def process_image(row, image_folder, label_folder):

    relative_path = str(row["Path"]).replace("\\", "/")

    source_image = GTSRB_PATH / relative_path

    if not source_image.exists():
        print("Missing:", source_image)
        return

    # Open image to get dimensions
    image = Image.open(source_image)

    width, height = image.size

    # -----------------------------------------------------
    # GTSRB ROI
    # -----------------------------------------------------

    x1 = float(row["Roi.X1"])
    y1 = float(row["Roi.Y1"])
    x2 = float(row["Roi.X2"])
    y2 = float(row["Roi.Y2"])

    # -----------------------------------------------------
    # Convert to YOLO format
    #
    # YOLO requires:
    # class center_x center_y width height
    #
    # All values normalized 0-1
    # -----------------------------------------------------

    center_x = ((x1 + x2) / 2) / width
    center_y = ((y1 + y2) / 2) / height

    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height

    class_id = int(row["ClassId"])

    # -----------------------------------------------------
    # Output filenames
    # -----------------------------------------------------

    image_name = source_image.name

    destination_image = image_folder / image_name

    label_name = source_image.stem + ".txt"

    destination_label = label_folder / label_name

    # -----------------------------------------------------
    # Copy image
    # -----------------------------------------------------

    shutil.copy2(
        source_image,
        destination_image
    )

    # -----------------------------------------------------
    # Write YOLO label
    # -----------------------------------------------------

    with open(
        destination_label,
        "w"
    ) as f:

        f.write(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}\n"
        )


# =========================================================
# PROCESS TRAINING DATA
# =========================================================

print("\nPreparing training images...")

for index, (_, row) in enumerate(train_df.iterrows()):

    process_image(
        row,
        images_train,
        labels_train
    )

    if (index + 1) % 1000 == 0:

        print(
            f"Processed {index + 1} "
            f"/ {len(train_df)}"
        )


# =========================================================
# PROCESS VALIDATION DATA
# =========================================================

print("\nPreparing validation images...")

for index, (_, row) in enumerate(val_df.iterrows()):

    process_image(
        row,
        images_val,
        labels_val
    )

    if (index + 1) % 1000 == 0:

        print(
            f"Processed {index + 1} "
            f"/ {len(val_df)}"
        )


# =========================================================
# CREATE DATASET YAML
# =========================================================

yaml_path = OUTPUT_PATH / "traffic_sign.yaml"

with open(yaml_path, "w") as f:

    f.write(
        f"""path: {OUTPUT_PATH.resolve().as_posix()}
train: images/train
val: images/val

nc: 43

names:
  0: Speed limit 20 km/h
  1: Speed limit 30 km/h
  2: Speed limit 50 km/h
  3: Speed limit 60 km/h
  4: Speed limit 70 km/h
  5: Speed limit 80 km/h
  6: End of speed limit 80 km/h
  7: Speed limit 100 km/h
  8: Speed limit 120 km/h
  9: No passing
  10: No passing for vehicles over 3.5 tons
  11: Right-of-way at intersection
  12: Priority road
  13: Yield
  14: Stop
  15: No vehicles
  16: Vehicles over 3.5 tons prohibited
  17: No entry
  18: General caution
  19: Dangerous curve left
  20: Dangerous curve right
  21: Double curve
  22: Bumpy road
  23: Slippery road
  24: Road narrows on the right
  25: Road work
  26: Traffic signals
  27: Pedestrians
  28: Children crossing
  29: Bicycles crossing
  30: Beware of ice/snow
  31: Wild animals crossing
  32: End of all speed and passing limits
  33: Turn right ahead
  34: Turn left ahead
  35: Ahead only
  36: Go straight or right
  37: Go straight or left
  38: Keep right
  39: Keep left
  40: Roundabout mandatory
  41: End of no passing
  42: End of no passing by vehicles over 3.5 tons
"""
    )

print("\n==========================================")
print("YOLO DATASET PREPARATION COMPLETE")
print("==========================================")
print(f"Dataset: {OUTPUT_PATH.resolve()}")
print(f"YAML:    {yaml_path.resolve()}")
print("==========================================")python prepare_yolo_dataset.py