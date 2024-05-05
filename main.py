import cv2
import mediapipe as mp
import NeuralNetwork.main as nn
import Data_preprocessor.main as hp
import torch
import pandas as pd


def load_model(model_class, path, n_in, n_hiddens, n_out):
    model = model_class(n_in, n_hiddens, n_out)
    model.load_state_dict(torch.load(path))

    return model


hiddens = [64, 32]
model = load_model(nn.Net, "Models/model_xd.pth", 18, hiddens, 6)
model.eval()


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

    prediction_text = "?"  # Default prediction text when no hands or low confidence
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            labeled_image, hand_data = label_points(labeled_image, hand_landmarks)

            # Predict
            hand_processor = hp.HandProcessor()
            x_data = hand_processor.get_point_data(hand_positions=[hand_data], label='?')[0]
            del x_data['label']

            x_data = pd.DataFrame([x_data])
            x_data = torch.tensor(x_data.values, dtype=torch.float32)
            model.eval()
            with torch.inference_mode():
                logits = model(x_data)  # Get the raw output logits from the model
                probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
                pred = torch.argmax(probabilities, dim=1)  # Find the predicted class with highest probability
                max_probs = probabilities.max(dim=1).values  # Get the maximum probability for each instance
                if max_probs.max() < 0.8:
                    prediction_text = '?'
                else:
                    prediction_text = chr(pred.item() + ord('A'))

            # Draw the prediction on the frame
            cv2.putText(labeled_image, f'Prediction: {prediction_text}, probability: {round(max_probs.max().item() * 100, 2)}%', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(labeled_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands', cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()

