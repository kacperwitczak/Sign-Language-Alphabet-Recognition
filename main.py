import cv2
import mediapipe as mp
import numpy as np



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hand_positions = []


def label_points(image, landmarks):
    hand_data = {}
    for idx, landmark in enumerate(landmarks.landmark):
        h, w, c = image.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        label = f'a{idx + 1}'
        cv2.putText(image, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        hand_data[label] = (landmark.x, landmark.y, landmark.z)
    return image, hand_data





while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    labeled_image = image.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            labeled_image, hand_data = label_points(labeled_image, hand_landmarks)
            hand_positions.append(hand_data)
            mp_drawing.draw_landmarks(labeled_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands', cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
        break

print(hand_positions)
print(len(hand_positions))



hands.close()
cap.release()
cv2.destroyAllWindows()
