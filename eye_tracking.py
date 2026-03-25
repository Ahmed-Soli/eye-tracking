import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGES_DIR = "images"
OUTPUT_DIR = "output"

def video_feed():
    cap = cv2.VideoCapture(index=0)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
        cv2.rectangle(img=frame, pt1=(50, 50), pt2=(200, 200), color=(0, 255, 0), thickness=8)
        cv2.putText(img=frame, text="Press 'q' to quit", org=(50, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        
        cv2.imshow(winname="Video Feed", mat=frame)

        if cv2.waitKey(delay=1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

def detect_eyes():
    os.makedirs(name=OUTPUT_DIR, exist_ok=True)

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
        eye_img = cv2.imread(filename=input_path)

        if eye_img is None:
            print(f"  [SKIP] Could not read: {filename}")
            continue

        eye_gray = cv2.cvtColor(src=eye_img, code=cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(src=eye_gray, thresh=30, maxval=255, type=cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(a=eye_img, dtype=np.uint8)
        cv2.drawContours(image=contour_image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2)

        name, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_contour.jpg")
        cv2.imwrite(filename=output_path, img=contour_image)
        print(f"  [OK] {filename} -> {output_path}")

    print(f"\nAll results saved to '{OUTPUT_DIR}/'")
    print("Displaying results...")

    for filename in files:
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_contour.jpg")
        result = cv2.imread(filename=output_path)
        if result is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(X=cv2.cvtColor(src=result, code=cv2.COLOR_BGR2RGB))
            plt.title(label=f"{filename}")
            plt.axis("off")

    plt.show()


def face_detection():
    face_cascade = cv2.CascadeClassifier(filename=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(filename=cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(index=0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(image=roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img=roi_color, pt1=(ex, ey), pt2=(ex + ew, ey + eh), color=(0, 255, 0), thickness=2)
                cv2.putText(img=frame, text="Press 'q' to quit", org=(50, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

        cv2.imshow(winname='Face Detection', mat=frame)
        if cv2.waitKey(delay=1) & 0xFF == ord('q'):
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
