import matplotlib.pyplot as plt
import numpy as np

import yolov5
from ensemble_boxes import *
import copy
from PIL import Image, ImageDraw, ImageFont

yolo_fast = yolov5.load('./yolo_on_6_fast.pt')

def return_fast_output(yolo_model, img):

    image = img.copy()

    results_yolo = yolo_model(img)

    try:
        boxes = results_yolo.pred[0][:, :4].tolist()
        scores = results_yolo.pred[0][:, 4].tolist()
        labels = results_yolo.pred[0][:, 5].tolist()
    except:
        boxes = []
        scores_yolo = []
        labels_yolo = []

    dic = {}
    for each in labels:
        if each not in dic.keys():
            dic[each] = (0,[])

    for i in range(len(labels)):
        score , box = dic[labels[i]]
        if score < scores[i]:
            dic[labels[i]] = (scores[i], boxes[i])

    # print(dic)
    return dic

def draw_bounding_boxes_pillow(image, dic):
    # Convert image to a PIL Image (if it's not already)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Create a Draw object
    draw = ImageDraw.Draw(image)

    # Try to load a custom font or fall back to the default font
    try:
        font = ImageFont.truetype("arial.ttf", 16)  # Load TTF font
    except IOError:
        font = ImageFont.load_default()

    # Iterate through the dictionary to draw bounding boxes
    for label, (score, box) in dic.items():
        # Unpack the box coordinates
        x_min, y_min, x_max, y_max = box

        # Draw the rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Add the label and score above the bounding box
        text = f"{int(label)}: {score:.2f}"
        text_size = draw.textsize(text, font=font)
        text_background = [x_min, y_min - text_size[1], x_min + text_size[0], y_min]
        draw.rectangle(text_background, fill="red")  # Background for better text visibility
        draw.text((x_min, y_min - text_size[1]), text, fill="white", font=font)

    return image


# Load the transformed image
transformed_image = Image.open(transformed_image_path)
# Perform detection and inference
transformed_img=np.array(transformed_image)
# detection_dict = number_detection(transformed_img, mode)
temp = return_fast_output(yolo_fast, transformed_img)
box_img = draw_bounding_boxes_pillow(transformed_image, temp)
# Save the image in JPG format
box_img.save("output_image.jpg", format="JPEG")
print("Image saved as 'output_image.jpg' in the current directory.")



