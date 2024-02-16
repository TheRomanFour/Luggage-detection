import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("nis od tega")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 35 == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)

    cap.release()

if __name__ == "__main__":
    video_path = r"D:\Projekti\Luggage detection\dizlederdof aedrom.mp4"
    output_folder = r"D:\Projekti\Luggage detection\frames"

    extract_frames(video_path, output_folder)
