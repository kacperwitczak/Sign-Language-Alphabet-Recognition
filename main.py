import cv2
import mediapipe as mp
import numpy as np

pairs_to_calculate = [['a1', 'a5'], ['a1', 'a9'], ['a1', 'a13'], ['a1', 'a17'], ['a1', 'a21'],
                      ['a5', 'a9'], ['a5', 'a13'], ['a5', 'a17'], ['a5', 'a21'],
                      ['a9', 'a13'], ['a9', 'a17'], ['a9', 'a21'],
                      ['a13', 'a17'], ['a13', 'a21'],
                      ['a17', 'a21']]

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


def calculate_distances(hand_data):
    distances = {}
    for pair in pairs_to_calculate:
        point1, point2 = pair
        x1, y1, z1 = hand_data[point1]
        x2, y2, z2 = hand_data[point2]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        distances[f"{point1}:{point2}"] = distance
    return distances


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

for idx, hand_data in enumerate(hand_positions):
    distances = calculate_distances(hand_data)
    distances_values = np.array(list(distances.values()))
    sum_of_distances = np.sum(distances_values)
    distances_normalized = {key: value / sum_of_distances for key, value in distances.items()}

    all_x = [v[0] for v in hand_data.values()]
    all_y = [v[1] for v in hand_data.values()]
    all_z = [v[2] for v in hand_data.values()]

    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    min_z, max_z = np.min(all_z), np.max(all_z)

    max_diff_x = (max_x - min_x)
    max_diff_y = (max_y - min_y)
    max_diff_z = (max_z - min_z)

    print(f"Hand {idx + 1}:")
    print(f"Largest difference - X: {max_diff_x:.2f}, Y: {max_diff_y:.2f}, Z: {max_diff_z:.2f}")
    for key, value in distances_normalized.items():
        print(f"{key}: {value:}")

hands.close()
cap.release()
cv2.destroyAllWindows()
