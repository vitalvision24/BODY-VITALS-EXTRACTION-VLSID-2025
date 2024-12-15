import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(img):
    """
    Preprocesses the input image: denoise, convert to grayscale, and apply thresholding.
    Returns the binary image and its dimensions.
    """
    # Upcaling the image slightly for better resolution
    img_resized = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_LINEAR)

    # Applying median blur to reduce noise while preserving edges
    blurred = cv.medianBlur(img_resized, 5)

    # Converting to grayscale
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    # Applying thresholding for optimal binary separation
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return thresh, gray.shape

def extract_time_series(thresh):
    """
    Extract the time series signal from the binary image by dynamically isolating connected components.
    """
    Npixels_rate, Npixels_time = thresh.shape

    # Summing along rows to find signal intensity profile
    row_sums = np.sum(thresh == 255, axis=1)

    # Detecting the region of interest (ROI) dynamically
    signal_region = np.where(row_sums > row_sums.max() * 0.2)[0]
    if len(signal_region) == 0:
        raise ValueError("No significant signal region found in the binary image.")

    min_index, max_index = signal_region[0], signal_region[-1]

    # Cropping the binary image to isolate the ROI
    thresh_crop = thresh[min_index:max_index + 1, :]

    return thresh_crop

def compute_time_series(thresh_crop):
    """
    Compute the time series signal from the cropped binary image.
    """
    Npixels_rate, Npixels_time = thresh_crop.shape
    s1rate_pixels = [
        np.mean(np.where(thresh_crop[:, col] == 255)[0]) if np.any(thresh_crop[:, col] == 255) else np.nan
        for col in range(Npixels_time)
    ]

    # Filling gaps (NaNs) in the signal using linear interpolation
    s1rate_pixels = np.array(s1rate_pixels)
    nan_mask = np.isnan(s1rate_pixels)
    s1rate_pixels[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), s1rate_pixels[~nan_mask])

    return s1rate_pixels, Npixels_time

def save_time_series(pixels, s1rate, save_path=None):
    """
    Saves the extracted and scaled time series data figure
    """
    plt.figure(figsize=(12, 3.8))
    plt.plot(pixels, s1rate, 'k', linewidth=2)
    plt.gca().invert_yaxis()
    plt.title("Scaled Time Series Signal of HR", size=16)
    plt.xlabel('Time (sec)', size=14)
    plt.ylabel('Heart Rate Voltage (mV)', size=14)
    plt.grid()

    # Saving the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def digitize(img):
    """
    Main function to digitize the input image and extract the time series signal.
    """
    # Preprocessing the image
    thresh, (Npixels_rate, Npixels_time) = preprocess_image(img)

    # Extracting the time series region
    thresh_crop = extract_time_series(thresh)

    # Computing the time series signal
    s1rate_pixels, Npixels_time = compute_time_series(thresh_crop)

    # Scaling the signal
    s1rate = s1rate_pixels * 0.2

    # Converting time from pixels to seconds
    pixels = np.arange(len(s1rate_pixels)) * 0.02

    # Visualizing and save the scaled time series
    save_time_series(pixels, s1rate, save_path="digitized_heart_graph.png")

if __name__=="__main__":
    # Loading the image using OpenCV
    image_path = r"C:\Users\soham\Downloads\hgraph\image_40.jpg"
    img = cv.imread(image_path)

    # Checking if the image is loaded correctly
    if img is None:
        print(f"Error: Could not load image from {image_path}")
    else:
    # Calling the digitize function
        digitize(img)
