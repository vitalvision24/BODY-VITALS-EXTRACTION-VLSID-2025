import os
import numpy as np
from ensemble_boxes import *
import copy
from PIL import Image, ImageDraw, ImageFont
import paddleocr
from paddleocr import PaddleOCR, draw_ocr
from ultralytics import YOLO

# Loading YOLOv11 model for screen segmentation
yolo_seg = YOLO("/content/BODY-VITALS-EXTRACTION-VLSID-2025/models/yolov11m-seg-best.pt")
# Loading YOLOv11 model for data segmentation from the screen 
yolo_det = YOLO('/content/BODY-VITALS-EXTRACTION-VLSID-2025/models/yolov11m-det-best.pt')
# Loading PaddleOCR model to read data from the results of data segmentation
paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', ocr_version = 'PP-OCR', structure_version = 'PP-Structure',
                rec_algorithm = 'CRNN', max_text_length = 200, use_space_char = False, lan = 'en', det = False,
                cpu_threads = 12, cls = False,use_gpu=False )


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
      
  # Extracting the first prediction result assuming that it is the main screen
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
This function is used to segment various data from the screen. It takes yolo model and
image as input.
'''
def return_output(yolo_model, img):
  
  image = img.copy()

  # Get predictions from the YOLOv11 model
  result = yolo_model(img)

  try:
      # YOLOv11 format: Extract boxes, scores, and labels
    for r in result:
      boxes = r.boxes.xyxy.tolist()  # List of bounding boxes
      scores = r.boxes.conf.tolist()  # List of confidence scores
      labels = r.boxes.cls.tolist()  # List of class indices
      break
        
  except AttributeError:
      # Handle case when predictions are empty or missing
    return {}

  # Iterate through predictions and keep the highest-scoring box for each label
  dic = {}
  for each in labels:
    if each not in dic.keys():
      dic[each] = (0,[])

  for i in range(len(labels)):
    score , box = dic[labels[i]]
    if score < scores[i]:
      dic[labels[i]] = (scores[i], boxes[i])

  print("Dictionnary with Labels and Confidence Scores and Bounding Box Coordinates: ", dic)
  return dic

'''
This method is used for reading the data from the boxes made after detecting the data
'''
def recognize(image, dic, rec):
  
  vitals = {}
  labels = {0.0:'DBP', 1.0:'HR', 2.0:'HR_W', 3.0:'MAP', 4.0:'RR' , 5.0:'SBP', 6.0:'SPO2'}
  for each in dic.keys():
    if each==2.0:
      continue
    else:
      score, box = dic[each]
      xmin = int(box[0])
      xmax = int(box[2])
      ymin = int(box[1])
      ymax = int(box[3])
    print(xmin,xmax,ymin,ymax)
    img = image[ymin:ymax,xmin:xmax]
    text = rec.ocr(img,cls = False,det = False)[0][0][0]
    text = text.replace('(','').replace(')','').replace('/','').replace('-','').replace('*','')
    if text.isdigit():
      vitals[labels[each]] = text

  return vitals

"""
Draws screen box, label, and score on the image using PIL.
"""
def draw_screen_boxes_pillow(image, dic):
  

  # Converting image to a PIL Image (if it's not already)
  if not isinstance(image, Image.Image):
    image = Image.fromarray(image)

  # Creating a Draw object
  draw = ImageDraw.Draw(image)

  # Trying to load a custom font or fall back to the default font
  try:
    font = ImageFont.truetype("arial.ttf", 16) 
  except IOError:
    font = ImageFont.load_default()

  # Iterating through the dictionary to draw bounding boxes
  for label, (score, box) in dic.items():
    if not box:  
      continue

    # Unpacking the box coordinates
    x_min, y_min, x_max, y_max = map(int, box)

    # Drawing the rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    # Adding the label and score above the bounding box
    text = f"{label}: {score:.2f}"
    text_bbox = font.getbbox(text)  # Use getbbox to calculate text size
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

'''
This method draws bounding boxes for the screen detected by the model and returns the image
with the bounding box drawn around it.
'''
def draw_bounding_boxes_pillow(image, dic):
      

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
      
    # Unpacking the box coordinates
    x_min, y_min, x_max, y_max = box

    # Drawing the rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    # Adding the label and score above the bounding box
    text = f"{int(label)}: {score:.2f}"
    # Useing getbbox to calculate text size
    text_bbox = font.getbbox(text)  
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_background = [x_min, y_min - text_height, x_min + text_width, y_min]
    # Background for better text visibility
    draw.rectangle(text_background, fill="red") 
    draw.text((x_min, y_min - text_height), text, fill="white", font=font)

  return image

"""
Draw bounding boxes on the image with labels and values from number_dict.
Save the modified image with annotations.
"""
def draw_bounding_boxes_with_labels(image, dic, number_dict):
 

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

  # Labels dictionary (to map numerical labels to categories)
  labels = {0.0: 'DBP', 1.0: 'HR', 2.0:'HR_W', 3.0: 'MAP', 4.0: 'RR', 5.0: 'SBP', 6.0: 'SPO2'}

  # Iterating through bounding boxes and draw them
  for label, (score, box) in dic.items():
    x_min, y_min, x_max, y_max = [int(coord) for coord in box]

    # Drawing the bounding box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    # Adding the label and corresponding value from number_dict
    category = labels.get(label, "Unknown")
    value = number_dict.get(category, "N/A")  # Fetch value for the label
    text = f"{category}: {value}"

    text_bbox = font.getbbox(text)  # Use getbbox to calculate text size
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_background = [x_min, y_min - text_height -5, x_min + text_width, y_min]

    # Drawing text background and text
    draw.rectangle(text_background, fill="red")  # Background for better visibility
    draw.text((x_min, y_min - text_height - 5), text, fill="white", font=font)

  return image
    
"""
Save the results dictionary to a text file.
"""
def save_results_to_txt(results, output_file="results.txt"):

  with open(output_file, "w") as file:
    for img_name, values in results.items():
      file.write(f"Image: {img_name}\n")
      for label, value in values.items():
        file.write(f"{label}: {value}\n")
      file.write("\n")  # Add a blank line between entries
  print(f"Results saved to '{output_file}'")

'''
This function takes the input as the input image path. It calls various functions to do processing
of various types on the image and run models on it. It gives a dictionary as an output.
'''
def final_detection(image_path):

  # intializing results as dictionary
  results = {}
  
  img_name = os.path.basename(image_path)

  # Load the input image
  image = Image.open(image_path)
  
  # calling detect_screen method to detect the screen
  screen_dic = detect_screen(image_path)
  print(screen_dic)
  
  # Drawing bounding boxes and saving the result
  result_img = draw_screen_boxes_pillow(image, screen_dic)
  # Saving the image in the current directory for reference
  result_img.save("boxed_screen.jpg", format="JPEG")
  
  # Extracting the bounding box
  for _, (score, box) in screen_dic.items():
    if not box:  # If no box is detected, print a message and return
      print("No bounding box detected.")
      return None

  # Cropping and saving the bounding box part of the image
  transformed_image_path = crop_and_save_box(image, box)
  transformed_image = Image.open(transformed_image_path)
  transformed_img=np.array(transformed_image)

  # segmenting the data from the segmentted screen
  temp = return_output(yolo_det, transformed_img)
  box_img = draw_bounding_boxes_pillow(transformed_image, temp)
  # saving the image containing bounding boxes around various data on the screen
  box_img.save("output_image.jpg", format="JPEG")
  print("Image saved as 'output_image.jpg' in the current directory.")
  
  number_dict = recognize(transformed_img, temp, paddle_ocr)
  result_image = draw_bounding_boxes_with_labels(transformed_img, temp, number_dict)
  result_image.save("annotated_image.jpg")
  print("Image saved as 'annotated_image.jpg' in the current directory.")
  results[img_name] = number_dict
  
  return results


image_path = '/content/prashant_icu_mon--16_2023_1_1_22_18_54.jpeg'
results_dict = final_detection(image_path)
save_results_to_txt(results_dict, output_file="detection_results.txt")
