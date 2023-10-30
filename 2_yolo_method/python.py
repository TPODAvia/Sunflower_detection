import os
import cv2
import csv

def write_to_file(img_path, bbox):
    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    txt_name = os.path.join("/home/gumich/Sunflower_detection/test/", 'name.txt')
    with open(txt_name, 'a') as f:
        f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


with open('/home/gumich/Sunflower_detection/annotations.csv', 'r') as file:
    data = csv.reader(file)

    for item in data:

        img_path = item[0]
        ymin = int(item[1])
        xmin = int(item[2])
        ymax = int(item[3])
        xmax = int(item[4])
        
        # Get image dimensions
        # im = cv2.imread(img_path)
        w= 5472
        h= 3078

        # Convert bounding box format
        bbox = convert((w,h), [xmin, xmax, ymin, ymax])

        # Write to file
        write_to_file("/home/gumich/Sunflower_detection/test/", bbox)