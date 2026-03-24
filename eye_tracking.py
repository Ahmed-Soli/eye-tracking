import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGES_DIR = "images"
OUTPUT_DIR = "output"

def video_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 8)
        cv2.putText(frame, "Press 'q' to quit", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

def detect_eyes():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(supported_ext)
    )

    if not files:
        print(f"No images found in '{IMAGES_DIR}/'")
        return

    print(f"Found {len(files)} image(s) in '{IMAGES_DIR}/'")

    for filename in files:
        input_path = os.path.join(IMAGES_DIR, filename)
        eye_img = cv2.imread(input_path)

        if eye_img is None:
            print(f"  [SKIP] Could not read: {filename}")
            continue

        eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(eye_img, dtype=np.uint8)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        name, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_contour.jpg")
        cv2.imwrite(output_path, contour_image)
        print(f"  [OK] {filename} -> {output_path}")

    print(f"\nAll results saved to '{OUTPUT_DIR}/'")
    print("Displaying results...")

    for filename in files:
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_contour.jpg")
        result = cv2.imread(output_path)
        if result is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title(f"{filename}")
            plt.axis("off")

    plt.show()


def face_detection():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Eye Tracking - HCI Project")
    print("-" * 30)
    print("1. Video Feed")
    print("2. Pupil/Iris Detection")
    print("3. Face & Eye Detection")
    print("-" * 30)

    choice = input("Select mode (1-3): ").strip()

    if choice == "1":
        video_feed()
    elif choice == "2":
        detect_eyes()
    elif choice == "3":
        face_detection()
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
