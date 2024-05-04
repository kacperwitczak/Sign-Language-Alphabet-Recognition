import os
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    #funkcja ta dziala dla jednej klatki
    #oznaczanie punktow, nadanie im nazw a1,a2, itp
    #zapisywanie do hand_data punktow w postaci klucz:wartosc - jest to ich polozenie wzgledem punktu 0,0 w lewym gornym rogu
    def label_points(self, image, landmarks):
        hand_data = {}
        for idx, landmark in enumerate(landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            label = f'a{idx + 1}'
            cv2.putText(image, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            hand_data[label] = (landmark.x, landmark.y, landmark.z)
        return image, hand_data


    #oznaczanie dloni na klatce, obecnie w klatce moze byc wiecej niz jedna dlon, chyba powinnismy to ograniczyc tylko do jednej dloni
    #ta funkcja znajduje punkty na klatce i przekazuje je do funkcji ktora je oznacza i zwraca
    def detect_hands(self, frame):
        results = self.hands.process(frame)
        hand_positions = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame, hand_data = self.label_points(frame, hand_landmarks)
                hand_positions.append(hand_data)
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame, hand_positions

    def close(self):
        self.hands.close()

class VideoProcessor:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.display = True

    #przetwarza filmik
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file:", video_path)
            return

        #pozycje dloni na wszystkich klatkach
        hand_positions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_with_hands, hand_data = self.hand_detector.detect_hands(frame)
            hand_positions.append(hand_data)

            if self.display:
                cv2.imshow('MediaPipe Hands', cv2.cvtColor(frame_with_hands, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        #w tym momencie w hand_positions znajduja sie wszystkie pozycje dloni w danym filmie
        #trzeba je teraz przetworzyc i zapisac do pliku
        #trzeba sprawdzic czy w danym elemencie hand_positions znajduje sie 21 elementow tzn, czy zaden punkt dloni nie zostal uciety np byl poza kamera

        #w tym momencie mamy punkty z jednego video z jednego folderu, chyba najlepiej najpierw przetworzyc wszystkie pliki osobno, tzn zapisywac np filmik1 z folderu A
        #do folderu Data_przetworzone/A_przetworzone/filmik1_przetworzone.csv
        #a potem polaczyc te wszystkie pliki csv w calosc
        #w csv powinno byc cos w tym stylu
        #label,a1a5(odleglosc od a1 do a5),...
        #A,0.5784,...
        #cos w tym formacie
        print(hand_positions)

        print("Processed: ", video_path)
        cap.release()

    #przetwarza wszystkie filmiki w danym folderze
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

#przetwarza wszystkie foldery w folderze Data
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

#przetwarzanie folderu
def process_folder(folder_path):
    video_processor = VideoProcessor()
    video_processor.process_videos_in_folder(folder_path)


def main():
    data_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
    if os.path.exists(data_folder_path):
        print("Processing videos in the specified folder:")
        process_folders_in_data(data_folder_path, multithreading=False)
    else:
        print("The specified folder does not exist.")


if __name__ == "__main__":
    main()
