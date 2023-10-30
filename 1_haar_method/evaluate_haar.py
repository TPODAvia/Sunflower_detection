import cv2
import csv
import numpy as np

w = 5472
h = 3078

img = np.zeros((h, w, 1), dtype=np.uint8)
img2 = np.zeros(img.shape[:2], dtype=np.uint8)

with open('/home/gumich/Sunflower_detection/annotations.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        ymin, xmin, ymax, xmax = map(int, row[1:])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

# Converting image to grayscale 
img_orig = cv2.imread('/home/gumich/Sunflower_detection/images/0000031830.jpg') 
gray_img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) 
haar_cascade = cv2.CascadeClassifier('/home/gumich/Sunflower_detection/1_haar_method/opencv_blog_content/classifier/cascade.xml') 
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 3) 

count_black = 0
# Iterating through rectangles of detected faces 
for (x, y, w, h) in faces_rect: 
    count_black += 1
    cv2.rectangle(img2, (x, y), (x+w, y+h), (255, 255, 255), -1) 

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
    
# cv2.imwrite('output_image.jpg', img)
# cv2.imwrite('output_image.jpg', img)