from ultralytics import YOLO
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────
DATA_YAML  = r"C:\avqc_project\data.yaml"
MODEL      = "yolov8n.pt"   # Sabse chhota aur fast model
EPOCHS     = 50             # Kitni baar dataset dekhega
IMG_SIZE   = 640            # Image size
BATCH      = 8              # Ek baar mein kitni images
PROJECT    = r"C:\avqc_project\runs"
NAME       = "pcb_defect_v1"
# ──────────────────────────────────────────────────────────

def train():
    print("🚀 PCB Defect Model Training Shuru!")
    print(f"   Model    : {MODEL}")
    print(f"   Epochs   : {EPOCHS}")
    print(f"   Image Size: {IMG_SIZE}")
    print(f"   Batch    : {BATCH}")
    print("-" * 40)

    # Model load karo
    model = YOLO(MODEL)

    # Training shuru karo
    results = model.train(
        data    = DATA_YAML,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        project = PROJECT,
        name    = NAME,
        device  = 0,        # 0 = GPU, 'cpu' = CPU
        workers = 2,
        patience= 20,       # Agar 20 epochs mein improve na ho toh ruk jao
        save    = True,
        plots   = True,
        verbose = True
    )

    print("\n✅ Training Complete!")
    print(f"📁 Results save hue: {PROJECT}\\{NAME}")
    print(f"🏆 Best model: {PROJECT}\\{NAME}\\weights\\best.pt")

if __name__ == "__main__":
    train()