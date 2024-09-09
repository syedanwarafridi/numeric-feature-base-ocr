import os
import json
import csv
import cv2
import numpy as np
import re

# Load JSON file
with open('filtered-data.json') as f:
    json_data = json.load(f)

# Keywords for categorization
personal_info_keywords = re.compile(r"(NAME|BIRTH|SEX|SEXO|M|GENDER|CITIZEN)", re.IGNORECASE)

def extract_features(json_data):
    features = []
    for key, value in json_data.items():
        if key != '0':  # Example to focus only on one key, adjust as needed
            continue
        path = value['path']
        type = value['type']
        angle = value['angle']
        size = value['size']
        yolo_result = value['yolo_result']
        ocr = value['ocr']
        
        # Extract OCR features (coordinates and categorize)
        for text, coordinates in ocr.items():
            x_coords = [coord['x'] for coord in coordinates]
            y_coords = [coord['y'] for coord in coordinates]
            
            # Determine category based on keywords
            category = "Unknown"
            if personal_info_keywords.search(text):
                category = "Personal Info"
                if text.upper() == "NAME" or text.upper() == "BIRTH":
                    category = "Personal Info"
            
            feature_vector = [
                path, 
                text, 
                x_coords,  # Original x coordinates
                y_coords,   # Original y coordinates
                category    # Category label
            ]
            features.append(feature_vector)
    
    return features

# Load the image
image_path = json_data['0']['path'] + ".jpeg"
image = cv2.imread(os.path.join('images', image_path))

# Global variables for mouse cropping
start_x, start_y = 0, 0
end_x, end_y = 0, 0
cropping = False
cropped_image = None

# Crop the image on mouse event
def crop_image(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, cropping, cropped_image

    # Retrieve the features passed via `param`
    features = param['features']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Record starting point of the crop
        start_x, start_y = x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            end_x, end_y = x, y
            # Draw rectangle while cropping
            img_copy = image.copy()
            cv2.rectangle(img_copy, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.imshow('Original Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # Record end point and finalize the crop
        end_x, end_y = x, y
        cropping = False
        cropped_image = image[start_y:end_y, start_x:end_x]
        cv2.imshow('Cropped Image', cropped_image)
        
        # Recalculate and normalize OCR coordinates after cropping
        recalculate_and_draw_coordinates(features, start_x, start_y, end_x - start_x, end_y - start_y, cropped_image)

# Adjust OCR coordinates based on cropping and normalize
def recalculate_and_draw_coordinates(features, crop_x, crop_y, crop_w, crop_h, cropped_image):
    for feature in features:
        x_coords = feature[2]  # Original x coordinates
        y_coords = feature[3]  # Original y coordinates
        
        # Normalize original coordinates to [0, 1] range
        img_w, img_h = image.shape[1], image.shape[0]
        norm_x_coords = [x * img_w for x in x_coords]
        norm_y_coords = [y * img_h for y in y_coords]
        
        # Adjust coordinates based on the crop position
        re_x_coords = [(x - crop_x) / crop_w for x in norm_x_coords]
        re_y_coords = [(y - crop_y) / crop_h for y in norm_y_coords]
        
        # Ensure coordinates are between 0 and 1
        re_x_coords = [min(max(0, x), 1) for x in re_x_coords]
        re_y_coords = [min(max(0, y), 1) for y in re_y_coords]
        
        # Update the feature list with recalculated coordinates
        feature[2] = re_x_coords
        feature[3] = re_y_coords

        # Draw rectangles on the cropped image
        draw_rectangles_on_cropped_image(cropped_image, re_x_coords, re_y_coords)

# Draw rectangles on the cropped image
def draw_rectangles_on_cropped_image(cropped_image, x_coords, y_coords):
    num_coords = len(x_coords)
    for i in range(0, num_coords, 4):  # Assuming each rectangle is defined by 4 points
        if i + 3 < num_coords:
            x1, y1 = int(x_coords[i] * cropped_image.shape[1]), int(y_coords[i] * cropped_image.shape[0])
            x2, y2 = int(x_coords[i+2] * cropped_image.shape[1]), int(y_coords[i+2] * cropped_image.shape[0])
            cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Cropped Image with Rectangles', cropped_image)

# Extract the features from the JSON data
features = extract_features(json_data)

# Display image and enable cropping with mouse
cv2.imshow('Original Image', image)

# Pass `features` as a parameter to the callback
cv2.setMouseCallback('Original Image', crop_image, {'features': features})

cv2.waitKey(0)
cv2.destroyAllWindows()

# Write recalculated and normalized coordinates to CSV
with open('features11.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path', 'word', 're_x_coords', 're_y_coords', 'category'])
    for feature in features:
        path = feature[0]
        word = feature[1]
        re_x_coords = feature[2]
        re_y_coords = feature[3]
        category = feature[4]
        writer.writerow([path, word, re_x_coords, re_y_coords, category])
