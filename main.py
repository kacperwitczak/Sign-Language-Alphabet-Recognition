import cv2
import torch
import pandas as pd
from Common.utils import load_model
from Common.hand_detector import HandDetector
from Common.hand_processor import HandProcessor
from NeuralNetwork.NeuralNet import Net


model = load_model(Net, "Models/model_xd.pth")
model.eval()

cap = cv2.VideoCapture(0)
hp = HandProcessor()
hd = HandDetector()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    labeled_image, hand_data = hd.detect_hands(image)

    prediction_text = "?"
    if hand_data is not None:
        x_data = hp.get_point_data(hand_positions=[hand_data], label='?')[0]
        del x_data['label']

        x_data = pd.DataFrame([x_data])
        x_data = torch.tensor(x_data.values, dtype=torch.float32)
        model.eval()
        with torch.inference_mode():
            logits = model(x_data)
            probabilities = torch.softmax(logits, dim=1)
            pred = torch.argmax(probabilities, dim=1)
            max_probs = probabilities.max(dim=1).values
            if max_probs.max() < 0.8:
                prediction_text = '?'
            else:
                prediction_text = chr(pred.item() + ord('A'))

        # Draw the prediction on the frame
        cv2.putText(labeled_image, f'Prediction: {prediction_text}, probability: {round(max_probs.max().item() * 100, 2)}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Hands', cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()