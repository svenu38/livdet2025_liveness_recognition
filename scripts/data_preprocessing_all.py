import os
import csv
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict

# === CONFIG ===
RAW_DIR = Path("/Users/venu/Documents/livdet2025_liveness_recognition/data/raw")
PROCESSED_DIR = Path("/Users/venu/Documents/livdet2025_liveness_recognition/data/processed")
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"
IMG_SIZE = (256, 256)

# === Ensure output folders exist ===
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

train_rows, val_rows, test_rows = [], [], []
counts = defaultdict(int)

def process_image(src_path, dest_path):
    try:
        img = Image.open(src_path).convert("L")
        img = img.resize(IMG_SIZE)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path.with_suffix(".png"))
        return True
    except Exception as e:
        print(f"❌ Failed to process {src_path}: {e}")
        return False

def get_label_and_spoof_from_path(path_parts):
    for i in reversed(range(len(path_parts))):
        name = path_parts[i].lower()
        if name in ["live", "alive"]:
            return "Live", ""
        elif name in ["fake", "spoof"]:
            if i + 1 < len(path_parts):
                return "Fake", path_parts[i + 1]
            else:
                return "Fake", "Unknown"
        elif name in ["gelatin", "silicone", "latex", "playdoh", "wood glue", "body double", "ecoflex", "modasil"]:
            return "Fake", path_parts[i]
    return "Unknown", ""

# === Traverse all image files ===
for img_path in RAW_DIR.rglob("*.*"):
    if img_path.suffix.lower() not in [".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png"]:
        continue

    parts = img_path.parts

    try:
        # Find "Fingerprint" and extract year from path
        idx_fp = parts.index("Fingerprint")
        year = parts[idx_fp - 1] if idx_fp > 0 else "Unknown"

        # Detect split using path
        if "Training" in parts:
            split = "Training"
        elif "Testing" in parts:
            split = "Testing"
        else:
            print(f"⚠️ Skipping {img_path} — could not identify split.")
            continue

        # Sensor name from parts just after 'Training' or 'Testing'
        idx_split = parts.index(split)
        sensor = "_".join([
            p for p in parts[idx_split + 1 : idx_split + 4]
            if p.lower() not in ["live", "alive", "fake", "spoof"]
        ])

        # Label and spoof_type
        label, spoof_type = get_label_and_spoof_from_path(parts)

        # Output filename and destination
        rel_name = f"{year}_{sensor}"
        filename = f"{rel_name}_{img_path.stem}.png"

        if split == "Training":
            dest = TRAIN_DIR / rel_name / filename
            train_rows.append([str(dest), sensor, year, label, spoof_type, "train"])
        else:
            dest = TEST_DIR / rel_name / filename
            test_rows.append([str(dest), sensor, year, label, spoof_type, "test"])

        if process_image(img_path, dest):
            counts[(split, label)] += 1

    except Exception as e:
        print(f"⚠️ Skipped {img_path}: {e}")

# === Create validation set ===
train_rows, val_rows = train_test_split(train_rows, test_size=0.1, random_state=42)

# === Copy validation images ===
for row in val_rows:
    src = Path(row[0])
    new_dest = VAL_DIR / src.relative_to(TRAIN_DIR)
    new_dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.open(src).save(new_dest)
        row[0] = str(new_dest)
    except Exception as e:
        print(f"❌ Failed to move val image {src}: {e}")

# === Save CSVs ===
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "sensor", "year", "label", "spoof_type", "set"])
        writer.writerows(rows)

write_csv(PROCESSED_DIR / "train.csv", train_rows)
write_csv(PROCESSED_DIR / "val.csv", val_rows)
write_csv(PROCESSED_DIR / "test.csv", test_rows)

# === Summary Output ===
print("\n✅ Preprocessing Complete!")
print(f" - Raw Image Count: {sum(counts.values())}")
print(f" - Train Images: {len(train_rows)}")
print(f" - Val Images:   {len(val_rows)}")
print(f" - Test Images:  {len(test_rows)}")
print(f" - Total Output: {len(train_rows) + len(val_rows) + len(test_rows)}")
print("📊 Breakdown by Dataset and Label:")
for key, value in sorted(counts.items()):
    print(f"   {key[0]} | {key[1]} → {value} images")
