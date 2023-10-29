import cv2
import csv
a = 0

with open('D:\\Coding_AI\\Sunflower_detection\\annotations.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        image_path = row[0]
        ymin, xmin, ymax, xmax = map(int, row[1:])
        # Crop the image using OpenCV
        img = cv2.imread(image_path)
        cropped_img = img[ymin:ymax, xmin:xmax]
        # Save the cropped image
        a = 1 + a
        cv2.imwrite('cropped_' + str(a) + '.jpg', cropped_img)