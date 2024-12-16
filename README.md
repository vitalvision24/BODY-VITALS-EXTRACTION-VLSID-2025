# BODY VITALS EXTRACTION
The project presents an advanced approach to patient monitoring in hospitals through image processing and edge computing. Monitoring a patient’s vitals is an important aspect in healthcare practices as it may enable detection of potential problems before they worsen. To be precise, this involves monitoring the patient's vital signs, including pulse rate, blood pressure, and oxygen level, likely to provide the necessary information on the general health and well-being of the particular patient. These measurements give a good picture of how healthy the patient is overall.

We suggest a three-step solution to address the challenge: Preprocessing, Vital Detection and Optical Character Recognition (OCR). Finally, we would address the digitization of the Heart Rate graph and which could help in the prediction of early onset of heart-attacks.

## APPROACH
The approach we choose to tackle the problem is, we first capture the image of the system in frame and detect the screen using YOLO segmentation model. The segmentation model will be fine-tuned for detecting the vital monitor by predicting a mask for the digital screen of the model. The image is cropped to the segmentation mask using the predictions of the model. Now, to localize the different vital signals on the segmented screen we train a YOLO model for object detection task on a custom annotated dataset. The main vital signals that we are focusing will be divided into the following classes/labels for the object detection model to classify them into:


0: Diastolic Blood Pressure (DBP)


1: Heart Rate (HR)


2: Heart Rate Graph (HR_W)


3: Mean Arterial Pressure (MAP)


4: Respiratory Rate (RR)


5: Systolic Blood Pressure (SBP)


6: SPO2

After the fine-tuned model detects the various vital signs that we are focusing on, we need to read the text from the corresponding vitals. For this we will use an Optical Character Recognition (OCR) model. Once we have the textual data we have the entire data in digital format. For the digitization of the Heart Rate Graph, we will use image processing with the help of OpenCV library and visualization using Matplotlib. The following steps are performed to obtain the digitized heart rate graph:

### Image Preprocessing:


•	The input image is upscaled for better resolution.


•	Noise is reduced using median blurring while preserving edges.


•	The image is converted to grayscale, and Otsu’s thresholding is applied to obtain a binary image separating the graph from the background.

### Region of Interest (ROI) Detection:


•	The binary image is analysed row-wise to calculate intensity sums.


•	Rows with significant signal intensity (above a threshold) are identified as the region containing the graph.


•	The ROI is dynamically cropped to isolate the heart rate graph.


### Graph Extraction:


•	For each column in the cropped binary image, the vertical positions of graph points are computed as the mean position of white pixels (255 intensity).


•	Missing data points are handled using linear interpolation to ensure continuity.


### Signal Scaling:


•	The extracted positions are scaled to represent heart rate voltage (mV).


•	Time values are calculated by mapping pixel indices to seconds based on the graph’s resolution.


### Visualization:


•	The scaled time series is plotted with the X-axis representing time (seconds) and the Y-axis showing heart rate voltage (mV).


•	The graph is inverted vertically to match the original orientation of the signal.

![file_2024-12-15_19 39 14](https://github.com/user-attachments/assets/975a35ce-7592-4886-a0a9-6f601c136ada)


## STEPS TO IMPLEMENT THE PIPELINE:

1) Clone the repo.


```
git clone https://github.com/vitalvision24/BODY-VITALS-EXTRACTION-VLSID-2025.git
```

2) Change the directory to `BODY-VITALS-EXTRACTION-VLSID-2025`


```
cd BODY-VITALS-EXTRACTION-VLSID-2025
```

3) Install all the dependencies/requirements.


```
pip install -r requirements.txt
```


4) Run one of the following pipelines to get the result.

   a. Paddle OCR Pipeline:


   ```
   python pipeline_paddle.py
   ```


   b. Parseq OCR Pipeline:


   ```
   python pipeline_parseq.py
   ```


5) To get the digitized Heart Rate graph:


   ```
   python hr_graph_digitization.py
   ``` 
## RESULTS


* The results from the segmentation model can be found [here](https://github.com/vitalvision24/BODY-VITALS-EXTRACTION-VLSID-2025/tree/main/Results/Screen_Segmentation).
   * A result is also shown below:


  <img src="https://github.com/vitalvision24/BODY-VITALS-EXTRACTION-VLSID-2025/blob/main/Results/Screen_Segmentation/Screen2_out.jpg" width=400 height=400>


* The results for data detection from the segmented image can be found [here](https://github.com/vitalvision24/BODY-VITALS-EXTRACTION-VLSID-2025/tree/main/Results/Data_Detection).
   * A result is also shown below:


   <img src="https://github.com/vitalvision24/BODY-VITALS-EXTRACTION-VLSID-2025/blob/main/Results/Data_Detection/result_4.jpg" width=400 height=400>



* The results for OCR from the data detected can be found [here](https://github.com/vitalvision24/BODY-VITALS-EXTRACTION-VLSID-2025/tree/main/Results/OCR).
   * A result is also shown below:


   <img src="https://github.com/vitalvision24/BODY-VITALS-EXTRACTION-VLSID-2025/blob/main/Results/OCR/result_4.jpg" width=400 height=400>


