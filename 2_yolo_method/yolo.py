from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator

model = YOLO('/home/gumich/Sunflower_detection/2_yolo_method/training_results/sunflower/weights/best.pt')
img = cv2.imread('/home/gumich/Sunflower_detection/images/0000031830.jpg')
results = model.predict(img)

for r in results:
    annotator = Annotator(img)
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        c = box.cls
        print(f'Box: {b}, Class: {model.names[int(c)]}')  # Print bbox in terminal
        annotator.box_label(b, model.names[int(c)])  # Annotate the image

img = annotator.result()

cv2.imwrite(filename="/home/gumich/Sunflower_detection/2_yolo_method/image.jpg", img=img)

# cv2.imshow('YOLO V8 Detection', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
