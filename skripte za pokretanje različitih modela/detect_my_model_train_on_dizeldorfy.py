import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('models\dizeldorft_airport_model.pt')

model = YOLO('models\dizeldorft_airport_model_v2.pt')

output_width = 640
output_height = 480

BACKPACK_CLASS = 0
PERSON_CLASS = 1  

cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(r'videji za treniranje i testiranje\dizlederdof aedrom.mp4')

cap = cv2.VideoCapture(r'videji za treniranje i testiranje\walking_on_airport.mp4')

cap = cv2.VideoCapture(r'videji za treniranje i testiranje\People walking at the airpor.mp4')

cap = cv2.VideoCapture(r'videji za treniranje i testiranje\korzo_short.mp4')


while True:

    ret, frame = cap.read()

    frame = cv2.resize(frame, (output_width, output_height))

    if not ret:
        break

    results = model.predict(frame, verbose=False)
    pred = results[0].boxes.data

    for det in pred:
        class_id, confidence, x_min, y_min, x_max, y_max = int(det[5]), det[4], det[0], det[1], det[2], det[3]

        if confidence > 0.5:
            if class_id == PERSON_CLASS:
                label = f"Person: {confidence:.2f}"
                color = (255, 0, 0)  # Red 
            else:
                label = f"{ 'Luggage' if class_id == BACKPACK_CLASS else 'Suitcase' }: {confidence:.2f}"
                color = (0, 0, 255) if class_id == BACKPACK_CLASS else (0, 255, 0)  
            thickness = 2
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()