import os
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

    def detect_hands(self, frame):
        results = self.hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def close(self):
        self.hands.close()

class VideoProcessor:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.display = False

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file:", video_path)
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_with_hands = self.hand_detector.detect_hands(frame)

            if self.display:
                cv2.imshow('MediaPipe Hands', cv2.cvtColor(frame_with_hands, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print("Processed: ", video_path)
        cap.release()

    def process_videos_in_folder(self, folder_path):
        if not os.path.exists(folder_path):
            print(f"The folder {folder_path} does not exist.")
            return

        video_files = [file for file in os.listdir(folder_path) if file.endswith(('.mp4', '.avi', '.mov'))]
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            self.process_video(video_path)

    def __del__(self):
        cv2.destroyAllWindows()
        self.hand_detector.close()

def process_folders_in_data(data_folder_path, multithreading=False):
    subfolders = [folder for folder in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, folder))]
    if multithreading:
        with ThreadPoolExecutor() as executor:
            for folder in subfolders:
                folder_path = os.path.join(data_folder_path, folder)
                executor.submit(process_folder, folder_path)
    else:
        for folder in subfolders:
            folder_path = os.path.join(data_folder_path, folder)
            process_folder(folder_path)


def process_folder(folder_path):
    video_processor = VideoProcessor()
    video_processor.process_videos_in_folder(folder_path)


def main():
    data_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
    if os.path.exists(data_folder_path):
        print("Processing videos in the specified folder:")
        process_folders_in_data(data_folder_path, multithreading=True)
    else:
        print("The specified folder does not exist.")


if __name__ == "__main__":
    main()
