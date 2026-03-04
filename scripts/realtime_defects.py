import cv2
from ultralytics import YOLO
from pathlib import Path
import time
MODEL_PATH = r"C:\avqc_project\runs\pcb_defect_v1\weights\best.pt"
CONFIDENCE = 0.25          # Kitne % sure hone pe detect kare
CAMERA_ID  = 0             # 0 = default webcam

COLORS = {
    "missing_hole"    : (0, 0, 255),      # Red
    "mouse_bite"      : (0, 165, 255),    # Orange
    "open_circuit"    : (0, 255, 255),    # Yellow
    "short"           : (255, 0, 0),      # Blue
    "spur"            : (255, 0, 255),    # Purple
    "spurious_copper" : (0, 255, 0),      # Green
}

def run_detection():
    print("🚀 Real-time PCB Defect Detection Shuru!")
    print("   'Q' dabao band karne ke liye")
    print("-" * 40)

    # Model load karo
    print("🧠 AI Model load ho raha hai...")
    model = YOLO(MODEL_PATH)
    print("✅ Model ready!")

    # Camera kholo
    print("📷 Camera khul raha hai...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("❌ Camera nahi khula! Check karo webcam connected hai ya nahi.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✅ Camera ready!")
    print("\n📺 Window mein dekho — defects dikh rahe honge!")

    # FPS calculate karne ke liye
    fps_time = time.time()
    fps      = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera se frame nahi aaya!")
            break

        # AI se detect karo
        results = model(frame, conf=CONFIDENCE, verbose=False)

        # Results draw karo
        defect_count = 0
        for result in results:
            for box in result.boxes:
                # Box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf            = float(box.conf[0])
                cls_id          = int(box.cls[0])
                cls_name        = model.names[cls_id]

               
                color = COLORS.get(cls_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        
                label = f"{cls_name} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
                cv2.putText(frame, label, (x1+2, y1-4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                defect_count += 1
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
        status_color = (0, 0, 255) if defect_count > 0 else (0, 255, 0)
        status_text  = f"DEFECT DETECTED: {defect_count}" if defect_count > 0 else "✓ OK - No Defects"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Frame dikhao
        cv2.imshow("PCB Defect Detection - Press Q to Quit", frame)

        # Q dabane pe band karo
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Detection band kar di!")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!")

if __name__ == "__main__":
    run_detection()
