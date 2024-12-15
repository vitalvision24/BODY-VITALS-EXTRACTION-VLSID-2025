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
import yolov5
from ensemble_boxes import *
import re
import torch.nn.functional as F
import copy
from ultralytics import YOLO

# loading the YOLOv_11 model
yolo_seg = YOLO("./yolov11m-seg-best.pt")
# loading YOLOv_11 model
yolo_model_det = yolov5.load('./final_yolo_weights.pt')
# loading OCR model
model_ocr = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()


def detect_screen(image_path):
'''
This function takes the input image path as input and uses YOLOv_11 to segment the screen from the image. 
It returns an output dictionary containing the co-ordinates of the bounding box bounding the screen, 
the confidence score and the label.
'''
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

def draw_screen_boxes_pillow(image, dic):
'''
This method draws bounding boxes for the screen detected by the model and returns the image
with the bounding box drawn around it.
'''
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

def crop_and_save_box(image, box, save_path="transformed_image.jpg"):
    """
    Crops the part of the image inside the bounding box and saves it.
    """
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
'''
This function is used to segment various data from the screen. It takes yolo model and
image as input.
'''
  image = img.copy()
  # passing the image in the model
  results_yolo = yolo_model(img)

  try:
    boxes = results_yolo.pred[0][:, :4].tolist()
    scores_yolo = results_yolo.pred[0][:, 4].tolist()
    labels_yolo = results_yolo.pred[0][:, 5].tolist()
  except:
    boxes = []
    scores_yolo = []
    labels_yolo = []
  boxes_yolo = []
  for box in boxes:
    boxes_yolo.append([box[0]/1280, box[1]/720, box[2]/1280, box[3]/720])
  result_box = boxes_yolo
  result_conf = scores_yolo
  result_label = labels_yolo
  return result_box, result_conf, result_label, img


def draw_bounding_boxes_pillow(image, boxes, scores, labels, save_path):
"""
This method is used to draw bounding boxes on the image and save it.
"""
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
    print(f"Image saved as {save_path}")


def recognize(image, boxes, scores):
'''
This method is used for running optical character recognization
'''
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
    print(procs)
    # Feed the concatenated tensor to the OCR model
    preds = model_ocr(procs)
    
    labels = inference_pred(preds)

    return labels, image, boxes, scores

def preproc_image(img):
'''
This method preprocesses the image according to the input format of the OCR model
'''
    img = Image.fromarray(img).convert('RGB')
    transform = T.Compose([
            T.Resize((32, 128)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
    img = transform(img)
    return img.unsqueeze(0)

def inference_pred(pred):
'''
This method is used to normalize the predicted results
'''
    pred = pred.softmax(-1)
    label, _ = model_ocr.tokenizer.decode(pred)
    return label


def wbf_ensemble(boxes_list, scores_list, labels_list, image):
  weights = [2, 1]
  iou_thr = 0.6
  skip_box_thr = 0.01
  sigma = 0.1
  boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
  return boxes, scores


def draw_bounding_boxes_with_labels(image, boxes, scores, text_labels):
    """
    This method draws bounding boxes on the image with labels and detected numbers.
    Display the detected data above the boxes.
    """

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
    image.save('output_image_with_values.jpg')
    print("Image saved as output_image_with_values.jpg")
    return image

def check_string(string):
  if(('(' in string or ')' in string) and '/' in string):
      return False
  elif (string.count('(') > 1 or string.count('/') > 1 or string.count(')') > 1):
      return False
  string = string.replace('(', '').replace('/', '').replace(')', '')
  pattern = r'^[\d]+$'
  return re.match(pattern, string) != None

def image_dict(text , boxes , scores, image):
'''
This method is used for making custom dictionary for the outputs that we received
'''
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

class CRABBNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=0, in_chans=4)
        self.ll = nn.Linear(2048+7,6)
        self.fl = nn.Linear(14,6)

    def forward(self, img, text_data, num, val_vec):
        x = self.model(img)
        x = torch.cat([x, num.view(-1, 1), self.fl(torch.cat([num.view(-1, 1), text_data.view(-1, 3), val_vec.view(-1, 10)], dim=1))], dim=1)
        x = self.ll(x)
        x = x.view(-1, 6)
#         x = F.softmax(x, dim = 1)

        return x

model_crabb = CRABBNET()
model_crabb = model_crabb.to(device)

'''
This is used for making custom dataset
'''
class InferDataset(Dataset):
    def __init__(self, img_dict):
        super().__init__()

        self.img_dict = img_dict

    def __getitem__(self, idx):
        img = self.img_dict['image']
        val_vec = np.array(self.img_dict['val_vec'])
        val_vec.resize(10,)
        np.random.shuffle(val_vec)

        boxes = self.img_dict['boxes'][idx]
        bbox = boxes['bbox']

        xmin = int(bbox[0]*1280)
        ymin = int(bbox[1]*720)
        xmax = int(bbox[2]*1280)
        ymax = int(bbox[3]*720)
        mask = np.zeros((img.shape[0], img.shape[1], 1))
        mask = cv2.rectangle((mask), (xmin, ymin), (xmax, ymax), 255, -1)

        # plt.imshow(mask[:,:,0])
        # plt.show()

        num = boxes['num']
        text_data = boxes['text_data']

        img = cv2.resize(img, (224, 224))
        mask = cv2.resize(mask, (224,224))[:,:,None]

        arr4 = np.concatenate([img, mask], axis = 2)

        arr4 = (np.transpose(arr4, (2, 0, 1))) / 255.0
        arr4 = torch.tensor(arr4)
        val_vec = torch.tensor(val_vec)

        return arr4, torch.tensor(num), torch.tensor(text_data), val_vec

    def __len__(self):
        return len(self.img_dict['boxes'])

def pred_organizing(pred_mat, nums):
'''
This method is used for making the output dictionary after getting all the outputs
'''
    num_boxes = nums.shape[0]
    box_store = set(range(num_boxes))
    mat_variance = torch.var(pred_mat, dim=0)
    argmax_dim0 = torch.argmax(pred_mat, dim=0)
    sorted_, indices = torch.sort(mat_variance, descending=True)

    res_dict = {'HR':None, 'RR':None, 'SPO2':None, 'SBP':None, 'DBP':None, 'MAP':None}
    label_cols = ['HR', 'RR', 'SPO2', 'SBP', 'DBP', 'MAP']

    for ind in indices:

        argmax_dim0 = torch.argmax(pred_mat, dim=0)
        box_ind = argmax_dim0[ind].item()

        if box_ind not in box_store:
            present = list(box_store)
            if len(present) == 0:
                continue

        res_dict[label_cols[ind]] = nums[box_ind]
        pred_mat[box_ind, :] = -100000

        if torch.unique(pred_mat).shape[0] == 1:
            break

    return res_dict

def final_inference(img_dict):
'''
This method is used to classify the data received from the OCR
'''
    test_img = InferDataset(img_dict)

    test_loader = DataLoader(
            dataset=test_img,
            batch_size=len(img_dict['boxes']),
            num_workers=0,
    )

    # Get the batch of data
    imgs, nums, text_data, val_vec = next(iter(test_loader))

    # Ensuring all tensors are on the same device as the model
    imgs = imgs.float().to(device)  # Move to GPU
    nums = nums.float().to(device)  # Move to GPU
    text_data = text_data.float().to(device)  # Move to GPU
    val_vec = val_vec.float().to(device)  # Move to GPU

    with torch.no_grad():
        # Performing inference
        label_preds = model_crabb(imgs, text_data, nums, val_vec)

    # Moving back to CPU if needed for further processing
    nums = nums.cpu().numpy()

    # Organizing predictions
    yerr_dict = pred_organizing(label_preds, nums)
    yerr_dict = {k: v for k, v in yerr_dict.items() if v is not None}
    return yerr_dict

def draw_bounding_boxes_with_values(image, detection_dict, output_dict):
    """
    Draw bounding boxes on the image with correct labels and detected values.
    """
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

    # Mapping the output labels from output_dict to the respective boxes from detection_dict
    labels_dict = output_dict
    boxes = detection_dict['boxes']

    # Preparing a list of bounding boxes and their respective text labels
    box_labels = []

    # Mapping each label to its respective bounding box
    for label_name, value in labels_dict.items():
        for box in boxes:
            # Match the num value to the detected label
            if box['num'] == value:  
                xmin, ymin, xmax, ymax = box['bbox']

                # Convert normalized box coordinates to pixel values
                xmin = int(xmin * 1280)
                ymin = int(ymin * 720)
                xmax = int(xmax * 1280)
                ymax = int(ymax * 720)

                box_labels.append((xmin, ymin, xmax, ymax, label_name, value))

    # Iterating through the box_labels to draw the boxes and labels
    for (xmin, ymin, xmax, ymax, label_name, value) in box_labels:
        # Drawing the bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        # Adding the label and value above the bounding box
        text = f"{label_name}: {value:.1f}"

        # Getting text size and adjust position
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Drawing background for text (to improve visibility)
        text_background = [xmin, ymin - text_height - 5, xmin + text_width, ymin]
        draw.rectangle(text_background, fill="red")
        draw.text((xmin, ymin - text_height - 5), text, fill="white", font=font)

    # Saving the modified image with bounding boxes and text
    image.save('annotated_image.jpg')
    print("Image saved as annotated_image.jpg")
    return image

def number_detection(img):
'''
This method is used to detect the data in the screnn and then optically recognizing them
'''
    # sending the image to YOLO model for detecting various vital signs on the screen
    boxes, scores, result_label, img = return_output(yolo_model_det, img)

    draw_bounding_boxes_pillow(img, boxes, scores, result_label, 'output_image.jpg')

    text , img, boxes , scores = recognize(img, boxes, scores)
    draw_bounding_boxes_with_labels(img, boxes, scores, text)

    number_dict = image_dict(text , boxes , scores, img)

    return number_dict


def save_results_to_txt(results, output_file="results.txt"):
    """
    Save the results dictionary to a text file.
    """
    with open(output_file, "w") as file:
        for img_name, values in results.items():
            file.write(f"Image: {img_name}\n")
            for label, value in values.items():
                file.write(f"{label}: {value}\n")
            # Adding a blank line between entries
            file.write("\n")  
    print(f"Results saved to '{output_file}'")


def final_detection(image_path):
'''
This function takes the input as the input image path. It calls various functions to do processing
of various types on the image and run models on it. It gives a dictionary as an output.
'''
    # intializing results as dictionary
    results = {}
    img_name = os.path.basename(image_path)

    # Load the input image
    image = Image.open(image_path)
    img=np.array(image)
    
    # Calling detect_screen method to detect the screen
    screen_dic = detect_screen(image_path)

    
    # Drawing bounding boxes and saving the result
    result_img = draw_screen_boxes_pillow(image, screen_dic)
    
    # Saving the image in the current directory for reference
    result_img.save("boxed_screen.jpg", format="JPEG")
    
    # Extracting the bounding box
    for _, (score, box) in screen_dic.items():
      # If no box is detected, print a message and return  
      if not box:  
        print("No bounding box detected.")
        return None

    # Cropping and saving the bounding box part of the image
    transformed_image_path = crop_and_save_box(image, box)
    transformed_image = Image.open(transformed_image_path)
    transformed_img=np.array(transformed_image)

    detection_dict = number_detection(transformed_img)
    print(detection_dict)
    output_dict = final_inference(detection_dict)
    print(output_dict)
    image_with_bboxes = draw_bounding_boxes_with_values(transformed_image, detection_dict, output_dict)
    # Storing the result for the current image
    results[img_name] = output_dict

    return results

image_path = './2.jpg'
results_dict = final_detection(image_path)
save_results_to_txt(results_dict, output_file="detection_results.txt")
