import cv2
import csv
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

w = 5472
h = 3078

img = np.zeros((h, w, 1), dtype=np.uint8)
img2 = np.zeros(img.shape[:2], dtype=np.uint8)

with open('/home/gumich/Sunflower_detection/annotations.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        ymin, xmin, ymax, xmax = map(int, row[1:])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

img2 = np.zeros(img.shape[:2], dtype=np.uint8)
model = YOLO('/home/gumich/Sunflower_detection/2_yolo_method/training_results/sunflower/weights/best.pt')
img_predict = cv2.imread('/home/gumich/Sunflower_detection/images/0000031830.jpg')
results = model.predict(img_predict)

count_black = 0
for r in results:
    boxes = r.boxes
    for box in boxes:
        count_black += 1
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        b = (int(b[1]), int(b[0]), int(b[3]), int(b[2]))  # convert to (ymin, xmin, ymax, xmax) format
        cv2.rectangle(img2, (b[1], b[0]), (b[3], b[2]), (255,255,255), -1)

cv2.imwrite('/home/gumich/Sunflower_detection/output_image3.jpg', img2)


# compute the bitwise AND using the mask
masked_img = cv2.bitwise_and(img, img, mask=img2)

# Initialize counters
count = 0
count_white = 0

# Open the CSV file
with open('/home/gumich/Sunflower_detection/annotations.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        ymin, xmin, ymax, xmax = map(int, row[1:])
        count += 1

        # Extract the rectangle from the masked image
        rect = masked_img[ymin:ymax, xmin:xmax]

        # Count the number of white pixels
        white_pixels = np.sum(rect == 255)

        # Compute the total area of the rectangle
        total_pixels = rect.size

        # If more than 80% of pixels are white, increment the counter
        if white_pixels / total_pixels > 0.8:
            count_white += 1

false_positive = count_black - count_white

print ("True Positive: " + str(count_white) + "/" + str(count))
print ("False Positive: " + str(false_positive) + "/" + str(count))
print ("False Negative: " + str(count_white) + "/" + str(count))


print ("Good result: " + str(count_white/count*100))
    
cv2.imwrite('output_image.jpg', masked_img)
# cv2.imwrite('output_image.jpg', img)