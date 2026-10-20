import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PERSON_MODEL_PATH = BASE_DIR / "yolo11n.pt"
PPE_MODEL_PATH = BASE_DIR / "models" / "buildvision-hazards-v0.3.pt"

print("=" * 55)
print(" SMART SAFETY SENTINEL - PERSON + PPE DETECTION")
print("=" * 55)

print("\nLoading person model...")
person_model = YOLO(str(PERSON_MODEL_PATH))
print("Person model loaded.")

print("\nLoading PPE model...")
if not PPE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"PPE model not found: {PPE_MODEL_PATH}\n"
        "Check that models/buildvision-hazards-v0.3.pt exists."
    )
ppe_model = YOLO(str(PPE_MODEL_PATH))
print("PPE model loaded.")
print("\nPPE model classes:")
for class_id, class_name in ppe_model.names.items():
    print(f"  {class_id}: {class_name}")


def choose_video():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select Construction Video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg"),
                   ("All files", "*.*")]
    )
    root.destroy()
    return path


def draw_label(frame, text, x, y, color):
    x, y = max(0, int(x)), max(20, int(y))
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - th - baseline - 6)
    right = min(frame.shape[1] - 1, x + tw + 8)
    bottom = min(frame.shape[0] - 1, y + 3)
    cv2.rectangle(frame, (x, top), (right, bottom), color, -1)
    cv2.putText(frame, text, (x + 4, y - baseline - 1), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def monitor_source(source, window_title):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print("Error: Unable to open the webcam or video.")
        capture.release()
        return

    print(f"\nMonitoring: {window_title}")
    print("Press Q or ESC in the video window to return to the menu.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Video ended or a frame could not be read.")
                break

            # Detect and track people.
            person_result = person_model.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                conf=0.35, imgsz=640, device="cpu", verbose=False
            )[0]

            # Detect PPE/hazards with the separately trained PPE model.
            ppe_result = ppe_model.predict(
                frame, conf=0.30, imgsz=640, device="cpu", verbose=False
            )[0]

            # Start with person boxes and tracking IDs.
            output = person_result.plot()
            ppe_count = 0

            if ppe_result.boxes is not None:
                ppe_count = len(ppe_result.boxes)
                for box in ppe_result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = ppe_result.names.get(class_id, str(class_id))
                    color = (0, 165, 255)
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                    draw_label(output, f"PPE: {class_name} {confidence:.2f}",
                               x1, y1, color)

            people_count = len(person_result.boxes) if person_result.boxes is not None else 0
            cv2.rectangle(output, (10, 8), (430, 75), (25, 25, 25), -1)
            cv2.putText(output, f"People detected: {people_count}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(output, f"PPE/hazard detections: {ppe_count}", (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_title, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    while True:
        print("\nSelect an option:")
        print("1. Live Webcam")
        print("2. Upload Construction Video")
        print("3. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            monitor_source(0, "Smart Safety Sentinel - Webcam + PPE")
        elif choice == "2":
            video_path = choose_video()
            if video_path:
                monitor_source(video_path, "Smart Safety Sentinel - Video + PPE")
            else:
                print("No video selected.")
        elif choice == "3":
            print("Exiting Smart Safety Sentinel...")
            break
        else:
            print("Invalid choice. Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
