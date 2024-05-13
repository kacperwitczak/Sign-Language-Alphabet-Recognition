import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.VALID_HAND_POINTS_NUMBER = 21

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

                #check if all data is present
                if len(hand_data) == self.VALID_HAND_POINTS_NUMBER:
                    hand_positions.append(hand_data)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # if no hand return None
        return frame, hand_positions[0] if hand_positions else None

    def __del__(self):
        self.hands.close()
