import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn, optim
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import timm
import segmentation_models_pytorch as smp
import imutils
from skimage.transform import ProjectiveTransform
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from sklearn.model_selection import train_test_split
import gc
import glob
import random
from ensemble_boxes import *
import re
import torch.nn.functional as F
import copy
from ultralytics import YOLO
from supabase import create_client, Client

# loading the YOLOv_11 model
yolo_seg = YOLO("./models/yolov11m-seg-best.pt")
# loading YOLOv_5 model
yolo_model_det = YOLO('./models/yolov11m-det-best.pt')
# loading OCR model
model_ocr = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()

SUPABASE_URL = "https://xrfohlthwpjmjtfgggvt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhyZm9obHRod3BqbWp0ZmdnZ3Z0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ0NjQxNDYsImV4cCI6MjA1MDA0MDE0Nn0.bh7EhIEjT_VhQFR9o4CSKkyVM8aSNpHFT6vci1VV8kg"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

'''
This function takes the input image path as input and uses YOLOv_11 to segment the screen from the image. 
It returns an output dictionary containing the co-ordinates of the bounding box bounding the screen, 
the confidence score and the label.
'''
def detect_screen(image_path):

    # sending image in the model for segmentation
    # output of the model is a list containing all the detected screens
    results_seg = yolo_seg.predict(source=image_path)

    # Checking if there are any predictions in the results
    if len(results_seg) == 0:
        print("No objects detected.")
        return {}

    # Extracting the first prediction result
    # Assuming one result in the list
    result = results_seg[0]

    # Extracting bounding boxes, scores, and labels
    # List of box coordinates
    boxes = result.boxes.xyxy.tolist()
    # List of confidence scores
    scores = result.boxes.conf.tolist()
    # Class names or IDs
    labels = result.names

    # If no boxes are detected
    if len(boxes) == 0:
        print("No boxes detected.")
        return {}

    try:
        # Assuming the first detected box
        box = boxes[0]
        score = scores[0]
        label = labels[0]

        # Storing the detection in a dictionary
        dic = {label: (score, box)}

    except Exception as e:
        print(f"Error extracting results: {e}")
        dic = {}

    return dic

'''
This method draws bounding boxes for the screen detected by the model and returns the image
with the bounding box drawn around it.
'''
def draw_screen_boxes_pillow(image, dic):

    # Converting image to a PIL Image (if it's not already)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Creating a Draw object
    draw = ImageDraw.Draw(image)

    # Trying to load a custom font or fall back to the default font
    try:
        font = ImageFont.truetype("arial.ttf", 16)  # Load TTF font
    except IOError:
        font = ImageFont.load_default()

    # Iterating through the dictionary to draw bounding boxes
    for label, (score, box) in dic.items():
        if not box:  # Skip if no box is detected
            continue


        # Unpacking the box coordinates
        x_min, y_min, x_max, y_max = map(int, box)

        # Drawing the rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Adding the label and score above the bounding box
        text = f"{label}: {score:.2f}"
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Drawing a background rectangle for better text visibility
        text_background = [x_min, y_min - text_height, x_min + text_width, y_min]
        draw.rectangle(text_background, fill="red")
        draw.text((x_min, y_min - text_height), text, fill="white", font=font)

        return image
"""
Crops the part of the image inside the bounding box and saves it.
"""
def crop_and_save_box(image, box, save_path="transformed_image.jpg"):

    # Converting image to a PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Ensuring the box coordinates are integers
    x_min, y_min, x_max, y_max = map(int, box)

    # Cropping the image using the bounding box
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    # Saving the cropped image
    cropped_image.save(save_path, format="JPEG")
    print(f"Cropped image saved as '{save_path}' in the current directory.")

    return save_path



def return_output(yolo_model, img):

  image = img.copy()

  # Get predictions from the YOLOv11 model
  result = yolo_model(img)

  image = img.copy()
# passing the image in the model
  results_yolo = yolo_model(img)

  try:
   for r in result:
        boxes = r.boxes.xyxy.tolist()  # List of bounding boxes
        scores = r.boxes.conf.tolist()  # List of confidence scores
        labels = r.boxes.cls.tolist()
        print("\n#############",boxes, scores, labels,"\n")

  except:
    boxes = []
    scores = []
    labels = []

  boxes_yolo=[]
  for box in boxes:
    boxes_yolo.append([box[0]/1280, box[1]/720, box[2]/1280, box[3]/720])
  result_box = boxes_yolo
  result_conf = scores
  result_label = labels
  print("\n TYPE:",type(result_box),"\n")
  print(result_box,result_conf,result_label)
  return result_box, result_conf, result_label, img


def draw_bounding_boxes_pillow(image, boxes, scores, labels, save_path):

    # Converting the image to a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Creating a drawing context on the image
    draw = ImageDraw.Draw(image)

    # Trying to load a custom font or fall back to the default font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Iterating through each bounding box and draw
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        label = labels[i]

        # Converting the coordinates from relative to absolute
        xmin, ymin, xmax, ymax = [int(coord * 1280) if i % 2 == 0 else int(coord * 720) for i, coord in enumerate(box)]

        # Drawing the bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        # Drawing the label and score above the box
        text = f"{label}: {score:.2f}"
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Drawing the background rectangle for the text
        text_background = [xmin, ymin - text_height, xmin + text_width, ymin]
        draw.rectangle(text_background, fill="red")
        draw.text((xmin, ymin - text_height), text, fill="white", font=font)

    # Saving the image with bounding boxes drawn
    image.save(save_path)
    # print(f"Image saved as {save_path}")

'''
This method is used for running optical character recognization
'''
def recognize(image, boxes, scores):

    imgs = []
    for box in boxes:
        xmin = int(box[0] * 1280)
        ymin = int(box[1] * 720)
        xmax = int(box[2] * 1280)
        ymax = int(box[3] * 720)
        img = image[ymin:ymax, xmin:xmax]
        imgs.append(img)

    # Preprocessing images and collect them into a list
    # procs is a list now
    procs = [preproc_image(img) for img in imgs]

    # Concatenate only once after ensuring procs is a list of tensors
    procs = torch.cat(procs, dim=0)
    # print(procs)
    # Feed the concatenated tensor to the OCR model
    preds = model_ocr(procs)

    labels = inference_pred(preds)

    return labels, image, boxes, scores


'''
This method preprocesses the image according to the input format of the OCR model
'''
def preproc_image(img):

    img = Image.fromarray(img).convert('RGB')
    transform = T.Compose([
            T.Resize((32, 128)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
    img = transform(img)
    return img.unsqueeze(0)

'''
This method is used to normalize the predicted results
'''
def inference_pred(pred):

    pred = pred.softmax(-1)
    label, _ = model_ocr.tokenizer.decode(pred)
    return label




"""
This method draws bounding boxes on the image with labels and detected numbers.
Display the detected data above the boxes.
"""
def draw_bounding_boxes_with_labels(image, boxes, scores, text_labels):

    # Converting image to a PIL Image (if it's not already)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Creating a Draw object
    draw = ImageDraw.Draw(image)

    # Trying to load a custom font or fall back to the default font
    try:
        font = ImageFont.truetype("arial.ttf", 16)  # Load TTF font
    except IOError:
        font = ImageFont.load_default()

    # Iterating through bounding boxes and draw them
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = [int(coord * 1280) if i % 2 == 0 else int(coord * 720) for i, coord in enumerate(box)]

        # Drawing the bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        # Adding the recognized text and score above the bounding box
        text = f"{text_labels[i]}: {scores[i]:.2f}"

        # Getting text size and adjust position
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Drawing background for text (to improve visibility)
        text_background = [xmin, ymin - text_height - 5, xmin + text_width, ymin]
        draw.rectangle(text_background, fill="red")
        draw.text((xmin, ymin - text_height - 5), text, fill="white", font=font)

    # Saving the modified image with bounding boxes and text
    # image.save('output_image_with_values.jpg')
    # print("Image saved as output_image_with_values.jpg")
    return image

def check_string(string):
  if(('(' in string or ')' in string) and '/' in string):
      return False
  elif (string.count('(') > 1 or string.count('/') > 1 or string.count(')') > 1):
      return False
  string = string.replace('(', '').replace('/', '').replace(')', '')
  pattern = r'^[\d]+$'
  return re.match(pattern, string) != None

'''
This method is used for making custom dictionary for the outputs that we received
'''
def image_dict(text , boxes , scores, image):

  c = 0
  text_l = []
  boxes_l = []
  scores_l = []
  for i, t in enumerate(text):
    if check_string(t):
      text_l.append(t)
      boxes_l.append(boxes[i])
      scores_l.append(scores[i])

  boxes_l, scores_l = np.array(boxes_l), np.array(scores_l)
  nums = np.array([float(txt.replace('(', '').replace('/', '').replace(')', '')) for txt in text_l])
  try:
    ind = np.argsort(scores_l)[-6:]
    scores_l = scores_l[ind]
    text_l = [text_l[x] for x in ind]
    boxes_l = boxes_l[ind]
    nums = nums[ind]
  except:
    pass
  boxes_dic = []
  for i,num in enumerate(text_l):
    bbxi = boxes_l[i]
    nm = nums[i]
    text_data = np.array([0.0, 0.0, 0.0])
    if '/' in num:
      text_data[0] = 1.0
    if '(' in num:
      text_data[1] = 1.0
    if ')' in num:
      text_data[2] = 1.0
    boxes_dic.append({'bbox': bbxi, 'num': nm, 'text_data': text_data})

  return {'image': image, 'val_vec': nums.tolist(), 'boxes': boxes_dic}

def draw_boxes_with_labels(image, boxes, labels, detected_texts):
    # Define the mapping of numeric labels to text labels
    label_mapping = {
        0.0: "DBP",
        1.0: "HR",
        2.0: "HR_W",
        3.0: "MAP",
        4.0: "RR",
        5.0: "SBP",
        6.0: "SPO2"
    }

    # Scale the box coordinates back to the original image dimensions
    scaled_boxes = [
        (
            int(box[0] * 1280),
            int(box[1] * 720),
            int(box[2] * 1280),
            int(box[3] * 720)
        )
        for box in boxes
    ]

    # Generate random colors for the boxes
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in boxes]

    # Initialize the result dictionary
    result_dict = {}

    # Iterate through the boxes, labels, and detected texts
    for box, label, detected_text, color in zip(scaled_boxes, labels, detected_texts, colors):
        # Map numeric label to its corresponding text label
        text_label = label_mapping.get(label, str(label))  # Default to the numeric label if not found

        # Replace detected_text with "N.A." for HR_W
        if text_label == "HR_W":
            detected_text = "N.A."
        else:
            # Attempt to convert the detected_text to a float
            try:
                detected_value = float(detected_text)
            except ValueError:
                detected_value = None  # Use None if conversion fails

            # Update the result dictionary with the detected value
            result_dict[text_label.lower()] = detected_value

        # Extract coordinates
        xmin, ymin, xmax, ymax = box

        # Draw the bounding box with the unique color
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=2)

        # Create the label text (mapped label + detected text)
        text = f"{text_label}: {detected_text}"

        # Calculate the text size
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=font_thickness)[0]

        # Background rectangle for the text
        text_x = xmin
        text_y = ymin - 5 if ymin - 5 > text_size[1] else ymin + text_size[1] + 5
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y + 2), color=color, thickness=-1)

        # Put the label text on the image
        cv2.putText(image, text, (text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 0), thickness=font_thickness)

    # Save the image with the bounding boxes and labels
    cv2.imwrite("Annotated.jpg", image)

    # Return the result dictionary
    return result_dict


'''
This method is used to detect the data in the screnn and then optically recognizing them
'''
def number_detection(img):

    # sending the image to YOLO model for detecting various vital signs on the screen
    boxes, scores, result_label, img = return_output(yolo_model_det, img)

    draw_bounding_boxes_pillow(img, boxes, scores, result_label, 'output_image.jpg')
    # draw_bounding_boxes_with_labels(img, boxes, scores, text)
    text , img, boxes , scores = recognize(img, boxes, scores)
    final_dict = draw_boxes_with_labels(img, boxes, result_label, text)
    

    number_dict = image_dict(text , boxes , scores, img)

    return final_dict

"""
Save the results dictionary to a text file.
"""
def save_results_to_txt(data_dict, output_file="results_dict.txt"):
    """
    Save a dictionary to a text file.

    Parameters:
    - data_dict: dict, the dictionary to save.
    - output_file: str, the path of the file to save the results to.
    """
    with open(output_file, "w") as file:
        for label, value in data_dict.items():
            # Write the label and value in each line
            file.write(f"{label}: {value}\n")
        # Adding a blank line for clarity if needed in future appends
        file.write("\n")
    
    # print(f"Dictionary saved to '{output_file}'")


def final_detection(input_folder, output_folder):
    """
    Process images from the input folder to detect and annotate,
    save results in the output folder, and add detection data to Supabase DB.

    Args:
        input_folder (str): Path to the folder containing images.
        output_folder (str): Path to the folder to save annotated images.

    Returns:
        dict: Results with image names as keys and detected data as values.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize results dictionary
    results = {}
    proc_time = []

    # Get all image files from the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for img_name in image_files:
        start_time = time.time()
        image_path = os.path.join(input_folder, img_name)

        # Load the input image
        print(f"Processing {img_name}...")
        image = Image.open(image_path)
        img_array = np.array(image)

        # Detect screen
        screen_dic = detect_screen(image_path)
        if not screen_dic:
            print(f"No screen detected in {img_name}. Skipping...")
            continue

        # Draw and save screen bounding boxes
        result_img = draw_screen_boxes_pillow(image, screen_dic)
        boxed_image_path = os.path.join(output_folder, f"boxed_{img_name}")
        result_img.save(boxed_image_path, format="JPEG")

        # Extract bounding box
        for _, (score, box) in screen_dic.items():
            if not box:
                print(f"No bounding box found in {img_name}. Skipping...")
                continue

        # Crop and save the bounding box
        cropped_image_path = crop_and_save_box(image, box)
        transformed_image = Image.open(cropped_image_path)
        transformed_img_array = np.array(transformed_image)

        # Detect numbers or other objects
        detection_dict = number_detection(transformed_img_array)

        time_taken = time.time() - start_time
        proc_time.append(time_taken)

        # Convert detection dictionary values to integers
        detection_dict = {key: int(value) for key, value in detection_dict.items()}
        detection_dict["id"] = "d2377dea-764a-47f2-badf-f9c306dc7218"
        # Insert detection data into Supabase DB
        try:
            supabase.table("patient_vitals").insert(detection_dict).execute()
            print(f"Inserted data for {img_name}: {detection_dict}")
        except Exception as e:
            print(f"Error inserting data for {img_name}: {e}")


        # Save results
        results[img_name] = detection_dict

    return results, proc_time

    def extract_frames(video_path, output_folder, interval_seconds=1):
    """
    Extract frames from a video at a specified time interval and save them to a folder.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where extracted frames will be saved.
        interval_seconds (int): Time interval between frames to extract (in seconds).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_interval = interval_seconds * fps  # Frames to skip
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame is at the required interval
        if frame_number % frame_interval == 0:
            frame_time = frame_number / fps  # Time in seconds
            # Use zero-padded numbers for filenames
            output_file = os.path.join(output_folder, f"frame_{saved_frame_count:04d}_{frame_time:.2f}s.jpg")
            cv2.imwrite(output_file, frame)
            print(f"Saved frame at {frame_time:.2f}s to {output_file}")
            saved_frame_count += 1

        frame_number += 1

    # Release the video capture
    cap.release()

    print(f"Extraction complete. {saved_frame_count} frames saved to {output_folder}.")


video_path = "./test_video.mp4"
extracted_frames_folder = "./extracted_frames"
extract_frames(video_path, extracted_frames_folder, interval_seconds=4)
annotated_frames_folder = "./annotated_frames"
results, process_time = final_detection(extracted_frames_folder, annotated_frames_folder)
for time in process_time:
  print("Time taken for one frame: ", time)
