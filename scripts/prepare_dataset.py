import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────
IMAGES_PATH = Path(r"C:\avqc_project\dataset_raw\PCB_DATASET\images")
ANNOT_PATH  = Path(r"C:\avqc_project\dataset_raw\PCB_DATASET\Annotations")
OUTPUT_PATH = Path(r"C:\avqc_project\dataset")
TRAIN_RATIO = 0.8
CLASSES     = ["missing_hole", "mouse_bite", "open_circuit",
               "short", "spur", "spurious_copper"]
# ──────────────────────────────────────────────────────────

def convert_annotation(xml_path, img_width, img_height):
    tree   = ET.parse(xml_path)
    root   = tree.getroot()
    labels = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip().lower().replace(" ", "_")
        if class_name not in CLASSES:
            continue
        class_id = CLASSES.index(class_name)
        bbox     = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        cx = ((xmin + xmax) / 2) / img_width
        cy = ((ymin + ymax) / 2) / img_height
        w  = (xmax - xmin) / img_width
        h  = (ymax - ymin) / img_height
        labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return labels

def prepare():
    # Folders banao
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            (OUTPUT_PATH / split / sub).mkdir(parents=True, exist_ok=True)

    # Pehle dekho images folder mein kya hai
    print("📂 Images folder check kar raha hoon...")
    img_folders = list(IMAGES_PATH.iterdir())
    print(f"   Mili folders/files: {[x.name for x in img_folders]}")

    # Saari images aur annotations dhundo
    pairs = []
    for img_path in IMAGES_PATH.rglob("*.jpg"):
        # Annotation dhundo
        xml_path = ANNOT_PATH / img_path.parent.name / img_path.with_suffix(".xml").name
        if not xml_path.exists():
            # Try direct match
            xml_path = ANNOT_PATH / img_path.with_suffix(".xml").name
        if xml_path.exists():
            pairs.append((img_path, xml_path))

    print(f"📊 Total images + annotations mili: {len(pairs)}")

    if len(pairs) == 0:
        print("\n❌ Koi pairs nahi mile!")
        print("📂 Annotations folder mein kya hai:")
        for item in ANNOT_PATH.iterdir():
            print(f"   📁 {item.name}")
        print("📂 Images folder mein kya hai:")
        for item in IMAGES_PATH.iterdir():
            print(f"   🖼  {item.name}")
        return

    # Shuffle aur split
    random.shuffle(pairs)
    t = int(len(pairs) * TRAIN_RATIO)
    v = t + int(len(pairs) * 0.1)

    splits = {
        "train" : pairs[:t],
        "val"   : pairs[t:v],
        "test"  : pairs[v:]
    }

    counts = {"train": 0, "val": 0, "test": 0}

    for split_name, split_pairs in splits.items():
        print(f"⏳ {split_name} process ho raha hai... ({len(split_pairs)} images)")
        for i, (img_path, xml_path) in enumerate(split_pairs):
            try:
                img    = Image.open(img_path).convert("RGB")
                w, h   = img.size
                labels = convert_annotation(xml_path, w, h)
                if not labels:
                    continue
                fname = f"{split_name}_{i:05d}.jpg"
                img.save(OUTPUT_PATH / split_name / "images" / fname)
                lbl_file = OUTPUT_PATH / split_name / "labels" / fname.replace(".jpg",".txt")
                with open(lbl_file, "w") as f:
                    f.write("\n".join(labels))
                counts[split_name] += 1
            except Exception as e:
                print(f"⚠️  Skipped {img_path.name}: {e}")

    print("\n✅ Dataset ready!")
    print(f"   🟢 Train : {counts['train']} images")
    print(f"   🟡 Val   : {counts['val']} images")
    print(f"   🔵 Test  : {counts['test']} images")

    # data.yaml banao
    yaml = f"""path: C:/avqc_project/dataset
train: train/images
val: val/images
test: test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(r"C:\avqc_project\data.yaml", "w") as f:
        f.write(yaml)

    print("\n📄 data.yaml ban gaya!")
    print("🎉 Phase 1 Complete! Model training ke liye ready!")

if __name__ == "__main__":
    prepare()