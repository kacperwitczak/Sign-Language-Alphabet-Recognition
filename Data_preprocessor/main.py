import os
import cv2
import mediapipe as mp


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
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.hand_detector = HandDetector()

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

            cv2.imshow('MediaPipe Hands', cv2.cvtColor(frame_with_hands, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_videos_in_folder_recursive(self, folder_path=None):
        if folder_path is None:
            folder_path = self.folder_path

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                self.process_videos_in_folder_recursive(item_path)
            elif os.path.isfile(item_path) and item_path.endswith(('.mp4', '.avi', '.mov')):
                print("Processing video:", item_path)
                self.process_video(item_path)

    def __del__(self):
        self.hand_detector.close()


def main():
    data_folder_path = "Data"
    if os.path.exists(data_folder_path):
        # Process videos in the specified folder
        print("Processing videos in the specified folder:")
        video_processor = VideoProcessor(data_folder_path)
        video_processor.process_videos_in_folder_recursive()
    else:
        print("The specified folder does not exist.")


if __name__ == "__main__":
    main()
