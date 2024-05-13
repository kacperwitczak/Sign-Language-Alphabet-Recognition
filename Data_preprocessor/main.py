import os
import cv2
from concurrent.futures import ThreadPoolExecutor, wait
import pandas as pd
from Common.hand_detector import HandDetector
from Common.hand_processor import HandProcessor

class VideoProcessor:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.hand_processor = HandProcessor()
        self.display = False
        self.output_folder = os.path.join("..\\Processed_Data")

    def _process_video(self, video_path):
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

            if hand_data:
                hand_positions.append(hand_data)

            if self.display:
                cv2.imshow('MediaPipe Hands', cv2.cvtColor(frame_with_hands, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        output_dir = os.path.join(self.output_folder, label)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + ".csv")

        hand_features = self.hand_processor.get_point_data(hand_positions, label)

        self._save_to_csv(hand_features, output_path)
        print("Processed and data saved:", output_path)

        cap.release()

    def _save_to_csv(self, hand_positions, output_path):
        if not hand_positions:
            print("No hand positions data to save.")
            return

        columns = list(hand_positions[0].keys())
        df = pd.DataFrame(hand_positions, columns=columns)
        df.to_csv(output_path, index=False)

    def process_videos_in_folder(self, folder_path):
        if not os.path.exists(folder_path):
            print(f"The folder {folder_path} does not exist.")
            return

        video_files = [file for file in os.listdir(folder_path) if file.endswith(('.mp4', '.avi', '.mov'))]
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            self._process_video(video_path)


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
    processed_data_path = os.path.join("..", "Processed_Data")

    combined_dataframe = pd.DataFrame()
    if os.path.exists(processed_data_path) and os.path.isdir(processed_data_path):
        subdirs = [os.path.join(processed_data_path, d) for d in os.listdir(processed_data_path) if
                   os.path.isdir(os.path.join(processed_data_path, d))]
        for subdir in subdirs:
            # Find all CSV files in the subdir and read each one
            csv_files = [os.path.join(subdir, file) for file in os.listdir(subdir) if file.endswith('.csv')]
            for csv_file_path in csv_files:
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
        process_folders_in_data(data_folder_path, multithreading=True)

        combine_csv()

    else:
        print("The specified folder does not exist.")
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
