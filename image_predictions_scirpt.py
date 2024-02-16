import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO(r'models\dizeldorft_airport_model.pt')


# Define class IDs
BACKPACK_CLASS = 0
PERSON_CLASS = 1

# Load the image
image_path = r'slike\dizlederdof aedrom slika.png'
image = cv2.imread(image_path)

# Perform inference with your YOLOv8 model
results = model.predict(image, verbose=False)
pred = results[0].boxes.data

# Check for backpacks, suitcases, handbags, and persons in the image
for det in pred:
    class_id, confidence, x_min, y_min, x_max, y_max = int(det[5]), det[4], det[0], det[1], det[2], det[3]

    if confidence > 0.5 and class_id in [BACKPACK_CLASS, PERSON_CLASS]:
        if class_id == BACKPACK_CLASS:
            label = f"Backpack: {confidence:.2f}"
            color = (255, 0, 0)  # Blue for backpacks
        elif class_id == PERSON_CLASS:
            label = f"Person: {confidence:.2f}"
            color = (255, 255, 0)  # Yellow for persons

        thickness = 2
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
        cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

# Save the result
save_path = r'slike\aiport_version1.png'
cv2.imwrite(save_path, image)
