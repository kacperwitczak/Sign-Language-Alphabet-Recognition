import os
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, wait
import pandas as pd
import numpy as np


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def label_points(self, image, landmarks):
        hand_data = {}
        for idx, landmark in enumerate(landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            label = f'a{idx + 1}'
            cv2.putText(image, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            hand_data[label] = (landmark.x, landmark.y, landmark.z)
        return image, hand_data

    def detect_hands(self, frame):
        results = self.hands.process(frame)
        hand_positions = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame, hand_data = self.label_points(frame, hand_landmarks)
                hand_positions.append(hand_data)
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Jeżeli nie ma dłoni, zwróć None
        return frame, hand_positions[0] if hand_positions else None

    def close(self):
        self.hands.close()


class VideoProcessor:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.display = False
        self.output_folder = os.path.join("..\\Processed_Data")

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file:", video_path)
            return

        hand_positions = []
        label = str(os.path.abspath(video_path)).split("\\")[-2]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_with_hands, hand_data = self.hand_detector.detect_hands(frame)
            # if hand_data:
            #    dict = hand_data[0]
            #    #dict['label'] = label
            #    hand_data = dict
            if hand_data:
                hand_positions.append(hand_data)

            if self.display:
                cv2.imshow('MediaPipe Hands', cv2.cvtColor(frame_with_hands, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        output_dir = os.path.join(self.output_folder, os.path.splitext(os.path.basename(video_path))[0])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'output.csv')

        hp = HandProcessor()
        hand_features = hp.get_point_data(hand_positions, label)

        self.save_to_csv(hand_features, output_path)
        print("Processed and data saved:", output_path)

        print("Processed: ", video_path)
        cap.release()

    def save_to_csv(self, hand_positions, output_path):
        if not hand_positions:
            print("No hand positions data to save.")
            return

        # Make column names from dictionary keys
        columns = list(hand_positions[0].keys())

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(hand_positions, columns=columns)

        # df = pd.DataFrame([pos for frame_data in hand_positions for pos in frame_data])
        df.to_csv(output_path, index=False)

        print("Data saved to:", output_path)

    def process_videos_in_folder(self, folder_path):
        if not os.path.exists(folder_path):
            print(f"The folder {folder_path} does not exist.")
            return

        video_files = [file for file in os.listdir(folder_path) if file.endswith(('.mp4', '.avi', '.mov'))]
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            self.process_video(video_path)

    def __del__(self):
        self.hand_detector.close()


class HandProcessor:
    def calculate_distances(self, hand_data):
        pairs_to_calculate = [['a1', 'a5'], ['a1', 'a9'], ['a1', 'a13'], ['a1', 'a17'], ['a1', 'a21'],
                              ['a5', 'a9'], ['a5', 'a13'], ['a5', 'a17'], ['a5', 'a21'],
                              ['a9', 'a13'], ['a9', 'a17'], ['a9', 'a21'],
                              ['a13', 'a17'], ['a13', 'a21'],
                              ['a17', 'a21']]
        distances = {}
        for pair in pairs_to_calculate:
            point1, point2 = pair
            x1, y1, z1 = hand_data[point1]
            x2, y2, z2 = hand_data[point2]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            distances[f"{point1}:{point2}"] = distance
        return distances

    def sum_distances_between_points(self, hand_data):
        pairs_to_calculate = [
            ['a1', 'a2'], ['a2', 'a3'], ['a3', 'a4'], ['a4', 'a5'],
            ['a1', 'a6'], ['a6', 'a7'], ['a7', 'a8'], ['a8', 'a9'],
            ['a6', 'a10'], ['a10', 'a11'], ['a11', 'a12'], ['a12', 'a13'],
            ['a10', 'a14'], ['a14', 'a15'], ['a15', 'a16'], ['a16', 'a17'],
            ['a14', 'a18'], ['a18', 'a19'], ['a19', 'a20'], ['a20', 'a21'],
            ['a1', 'a18']
        ]
        total_distance = 0
        for pair in pairs_to_calculate:
            point1, point2 = pair
            x1, y1, z1 = hand_data[point1]
            x2, y2, z2 = hand_data[point2]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            total_distance += distance
        return total_distance

    def get_point_data(self, hand_positions, label):
        output = []
        for idx, hand_data in enumerate(hand_positions):
            distances = self.calculate_distances(hand_data)
            sum_of_hand_distances = self.sum_distances_between_points(hand_data)
            distances_normalized = {k: v / sum_of_hand_distances for k, v in distances.items()}

            all_x = [v[0] for v in hand_data.values()]
            all_y = [v[1] for v in hand_data.values()]
            all_z = [v[2] for v in hand_data.values()]

            min_x, max_x = np.min(all_x), np.max(all_x)
            min_y, max_y = np.min(all_y), np.max(all_y)
            min_z, max_z = np.min(all_z), np.max(all_z)

            max_diff_x = (max_x - min_x) / sum_of_hand_distances
            max_diff_y = (max_y - min_y) / sum_of_hand_distances
            max_diff_z = (max_z - min_z) / sum_of_hand_distances

            distances_normalized['max_diff_x'] = max_diff_x
            distances_normalized['max_diff_y'] = max_diff_y
            distances_normalized['max_diff_z'] = max_diff_z

            distances_normalized['label'] = label

            output.append(distances_normalized)

        return output


def process_folders_in_data(data_folder_path, multithreading=False):
    subfolders = [folder for folder in os.listdir(data_folder_path) if
                  os.path.isdir(os.path.join(data_folder_path, folder))]

    futures = []
    if multithreading:
        with ThreadPoolExecutor() as executor:
            for folder in subfolders:
                folder_path = os.path.join(data_folder_path, folder)
                future = executor.submit(process_folder, folder_path)
                futures.append(future)
        wait(futures)
    else:
        for folder in subfolders:
            folder_path = os.path.join(data_folder_path, folder)
            process_folder(folder_path)


def process_folder(folder_path):
    video_processor = VideoProcessor()
    video_processor.process_videos_in_folder(folder_path)


def combine_csv():
    processed_data_path = os.path.join("..\\Processed_Data")

    combined_dataframe = pd.DataFrame()
    if os.path.exists(processed_data_path) and os.path.isdir(processed_data_path):
        subdirs = [os.path.join(processed_data_path, d) for d in os.listdir(processed_data_path) if
                   os.path.isdir(os.path.join(processed_data_path, d))]
        for subdir in subdirs:
            csv_file_path = next((os.path.join(subdir, file) for file in os.listdir(subdir) if file.endswith('.csv')),
                                 None)
            if csv_file_path:
                temp_df = pd.read_csv(csv_file_path)
                combined_dataframe = pd.concat([combined_dataframe, temp_df], ignore_index=True)
    else:
        print("Processed_Data directory does not exist.")
        return

    combined_csv_path = os.path.join(processed_data_path, 'combined_data.csv')
    combined_dataframe.to_csv(combined_csv_path, index=False)
    print("Combined CSV created at:", combined_csv_path)


def main():
    data_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
    if os.path.exists(data_folder_path):
        print("Processing videos in the specified folder:")
        process_folders_in_data(data_folder_path, multithreading=False)

        combine_csv()

    else:
        print("The specified folder does not exist.")


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
