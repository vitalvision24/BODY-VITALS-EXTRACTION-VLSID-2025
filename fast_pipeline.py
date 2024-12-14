import os
import numpy as np

import yolov11
import yolov5
from ensemble_boxes import *
import copy
from PIL import Image, ImageDraw, ImageFont

import paddleocr
from paddleocr import PaddleOCR,draw_ocr

from ultralytics import YOLO
yolo_seg = YOLO("/content/yolov11m-seg-best.pt")
yolo_fast = yolov5.load('./yolo_on_6_fast.pt')
paddle_fast = PaddleOCR(use_angle_cls=False, lang='en', ocr_version = 'PP-OCR', structure_version = 'PP-Structure',
                rec_algorithm = 'CRNN', max_text_length = 200, use_space_char = False, lan = 'en', det = False,
                cpu_threads = 12, cls = False,use_gpu=False )

def detect_screen(image_path):
    # Load the transformed image
    image = Image.open(image_path)
    img = np.array(image)
    print(img)

    image1 = img.copy()
    results_seg = yolo_seg.predict(source=image_path)

    # Check if there are any predictions in the results
    if len(results_seg) == 0:
        print("No objects detected.")
        return {}

    # Extract the first prediction result
    result = results_seg[0]  # Assuming one result in the list
    
    # Extract bounding boxes, scores, and labels
    boxes = result.boxes.xyxy.tolist()  # List of box coordinates
    scores = result.boxes.conf.tolist()  # List of confidence scores
    labels = result.names  # Class names or IDs

    # If no boxes are detected
    if len(boxes) == 0:
        print("No boxes detected.")
        return {}

    try:
        # Assuming the first detected box
        box = boxes[0]
        score = scores[0]
        label = labels[0]

        # Store the detection in a dictionary
        dic = {label: (score, box)}

    except Exception as e:
        print(f"Error extracting results: {e}")
        dic = {}

    return dic



def return_fast_output(yolo_model, img):

    image = img.copy()

    results_yolo = yolo_model(img)

    try:
        boxes = results_yolo.pred[0][:, :4].tolist()
        scores = results_yolo.pred[0][:, 4].tolist()
        labels = results_yolo.pred[0][:, 5].tolist()
    except:
        boxes = []
        scores = []
        labels = []

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

def recognize_fast(image,dic,rec):

    vitals = {}
    labels = {0.0: 'DBP' , 1.0:'HR' , 2.0:'HR_W' , 3.0:'MAP', 4.0:'RR' , 5.0:'SBP' , 6.0:'SPO2' }
    for each in dic.keys():
        score, box = dic[each]
        xmin = int(box[0])
        xmax = int(box[2])
        ymin = int(box[1])
        ymax = int(box[3])
        img = image[ymin:ymax,xmin:xmax]
        text = rec.ocr(img,cls = False,det = False)[0][0][0]
        text = text.replace('(','').replace(')','').replace('/','').replace('-','').replace('*','')
        if text.isdigit():
            vitals[labels[each]] = text

    return vitals


def draw_screen_boxes_pillow(image, dic):
    """
    Draws screen box, label, and score on the image using PIL.
    """
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
        if not box:  # Skip if no box is detected
            continue


        # Unpack the box coordinates
        x_min, y_min, x_max, y_max = map(int, box)

        # Draw the rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Add the label and score above the bounding box
        text = f"{label}: {score:.2f}"
        text_bbox = font.getbbox(text)  # Use getbbox to calculate text size
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw a background rectangle for better text visibility
        text_background = [x_min, y_min - text_height, x_min + text_width, y_min]
        draw.rectangle(text_background, fill="red")
        draw.text((x_min, y_min - text_height), text, fill="white", font=font)

        return image

def crop_and_save_box(image, box, save_path="transformed_image.jpg"):
    """
    Crops the part of the image inside the bounding box and saves it.
    """
    # Convert image to a PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Ensure the box coordinates are integers
    x_min, y_min, x_max, y_max = map(int, box)
    
    # Crop the image using the bounding box
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    
    # Save the cropped image
    cropped_image.save(save_path, format="JPEG")
    print(f"Cropped image saved as '{save_path}' in the current directory.")

    return save_path



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
        text_bbox = font.getbbox(text)  # Use getbbox to calculate text size
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_background = [x_min, y_min - text_height, x_min + text_width, y_min]
        draw.rectangle(text_background, fill="red")  # Background for better text visibility
        draw.text((x_min, y_min - text_height), text, fill="white", font=font)

    return image

def draw_bounding_boxes_with_labels(image, dic, number_dict):
    """
    Draw bounding boxes on the image with labels and values from number_dict.
    Save the modified image with annotations.
    """
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

    # Labels dictionary (to map numerical labels to categories)
    labels = {0.0: 'DBP', 1.0: 'HR', 2.0:'HR_W', 3.0: 'MAP', 4.0: 'RR', 5.0: 'SBP', 6.0: 'SPO2'}

    # Iterate through bounding boxes and draw them
    for label, (score, box) in dic.items():
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]

        # Draw the bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Add the label and corresponding value from number_dict
        category = labels.get(label, "Unknown")
        value = number_dict.get(category, "N/A")  # Fetch value for the label
        text = f"{category}: {value}"

        text_bbox = font.getbbox(text)  # Use getbbox to calculate text size
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_background = [x_min, y_min - text_height -5, x_min + text_width, y_min]

        # Draw text background and text
        draw.rectangle(text_background, fill="red")  # Background for better visibility
        draw.text((x_min, y_min - text_height - 5), text, fill="white", font=font)

    return image

def save_results_to_txt(results, output_file="results.txt"):
    """
    Save the results dictionary to a text file.

    Parameters:
        results (dict): Dictionary to be saved.
        output_file (str): Name of the text file to save the results.
    """
    with open(output_file, "w") as file:
        for img_name, values in results.items():
            file.write(f"Image: {img_name}\n")
            for label, value in values.items():
                file.write(f"{label}: {value}\n")
            file.write("\n")  # Add a blank line between entries
    print(f"Results saved to '{output_file}'")


def final_detection(image_path):
    results = {}
    img_name = os.path.basename(image_path)

    # Load the transformed image
    image = Image.open(image_path)
    img=np.array(image)
    print(img)

    screen_dic = detect_screen(image_path)
    print(screen_dic)
    # Draw bounding boxes and save the result
    result_img = draw_screen_boxes_pillow(image, screen_dic)
    result_img.save("boxed_screen.jpg", format="JPEG")
    # Extract the bounding box
    for _, (score, box) in screen_dic.items():
      if not box:  # If no box is detected, print a message and return
        print("No bounding box detected.")
        return None

    # Crop and save the bounding box part of the image
    transformed_image_path = crop_and_save_box(image, box)
    transformed_image = Image.open(transformed_image_path)
    transformed_img=np.array(transformed_image)
    
    temp = return_fast_output(yolo_fast, transformed_img)
    box_img = draw_bounding_boxes_pillow(transformed_image, temp)
    box_img.save("output_image.jpg", format="JPEG")
    print("Image saved as 'output_image.jpg' in the current directory.")
    
    number_dict = recognize_fast(transformed_img, temp, paddle_fast)
    result_image = draw_bounding_boxes_with_labels(transformed_img, temp, number_dict)
    result_image.save("annotated_image.jpg")
    print("Image saved as 'annotated_image.jpg' in the current directory.")
    results[img_name] = number_dict
    
    return results


image_path = './2.jpg'
results_dict = final_detection(image_path)
save_results_to_txt(results_dict, output_file="detection_results.txt")
