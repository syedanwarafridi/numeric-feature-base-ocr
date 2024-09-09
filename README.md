# OCR Image Processing

This project involves extracting OCR features from images, recalculating and normalizing coordinates based on user-defined cropping, and saving the results to a CSV file. The script uses OpenCV for image processing, NumPy for numerical operations, and regular expressions for categorization.

## Requirements

The project requires the following Python packages:
- `opencv-python`: For image processing and drawing rectangles on images.
- `numpy`: For numerical operations and coordinate normalization.
- `matplotlib`: For displaying images.
- `jupyter`: For running the Jupyter notebook.

## Setup Instructions

1. **Create and Activate a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install Dependencies**:
    - **Create `requirements.txt`**:
      Create a file named `requirements.txt` in the root directory of your project with the following content:
      ```
      opencv-python
      numpy
      matplotlib
      jupyter
      ```
    - **Install Using pip**:
      ```bash
      pip install -r requirements.txt
      ```

3. **Run the Jupyter Notebook**:
    - Launch Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Open the notebook named `feature_based_ocr.ipynb` and run the cells to execute the code.

4. **Prepare the Data**:
   - Ensure you have the `filtered-data.json` file in the working directory.
   - Place the images in the `images` folder. The script expects images to be named according to the `path` field in the JSON data.

## Script Overview

- **Loading Data**: Reads JSON data and extracts OCR features.
- **Mouse Cropping**: Allows the user to crop the image using mouse events.
- **Coordinate Recalculation**: Adjusts and normalizes OCR coordinates based on the crop.
- **CSV Export**: Saves the recalculated coordinates and other features to a CSV file.

## Notes

- Ensure that the image paths and JSON data are correctly formatted and consistent with each other.
- If additional features or modifications are required, you may need to adjust the code accordingly.
