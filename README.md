# Yeast Density Classification using OpenCV and ANN

This project uses computer vision techniques (OpenCV) and a simple Artificial Neural Network (ANN) to classify microscope images of *Saccharomyces Cerevisiae* (SC) and *Saccharomyces Pastorianus* (SP) based on their **cell density**.

**Hypothesis:** The classification is based on the following rule:

>   * **Saccharomyces Cerevisiae (SC) = Low Density** (Assigned Label 0)
>   * **Saccharomyces Pastorianus (SP) = High Density** (Assigned Label 1)

The script automatically processes images from training folders, trains a classifier, and then provides a prompt for you to classify new, unseen images.

  

-----

## Project Goal 🎯

The goal of this project is to automate the classification of two types of yeast (SC and SP) from bright-field microscope images. Instead of complex shape (morphological) analysis, this model tests the hypothesis that the two types can be distinguished by their **average cell density** within the field of view.

## How It Works: The Pipeline ⚙️

The script runs a full image processing and machine learning pipeline.

### 1\. Image Registration (ROI Finding)

Microscope images often have a circular field of view (ROI) that is not centered. This step finds the circle, centers it, and crops the image to focus only on the relevant area.

1.  **Find ROI (`findRoiCircle`):**

      * The image is converted to grayscale.
      * A strong **`GaussianBlur`** is applied to blur out the individual cells, leaving only the large, bright circular shape.
      * **`cv2.threshold`** with the `THRESH_OTSU` flag is used to automatically create a binary mask of the bright circle.
      * **`cv2.findContours`** with `RETR_EXTERNAL` finds the outline of this circle.
      * **`cv2.minEnclosingCircle`** calculates the precise center and radius of this circle.

2.  **Center & Crop (`centerAndCropImage`):**

      * An **Affine Transformation** (`cv2.warpAffine`) is used to mathematically "shift" the image so the detected circle's center moves to the true center of the image.
      * A circular mask is applied to black out the dark corners.
      * The image is cropped based on the circle's radius and the `ZOOM_FACTOR`.

### 2\. Image Segmentation (Cell Detection)

This is the most critical step for density analysis. It turns the cropped image into a binary (black and white) mask where white pixels represent yeast and black pixels represent the background.

1.  **Pre-processing:**

      * The cropped image is converted to grayscale.
      * **`cv2.medianBlur`** is applied to reduce small "salt-and-pepper" noise.
      * **`cv2.createCLAHE`** (Contrast Limited Adaptive Histogram Equalization) is applied to enhance local contrast, making cell edges more distinct, especially in poorly lit areas.

2.  **Thresholding:**

      * **`cv2.adaptiveThreshold`** is used instead of a simple global threshold. This is crucial for images with uneven lighting, as it calculates different thresholds for different regions of the image.
      * `THRESH_BINARY_INV` is used because the yeast cells are *darker* than the background, so this inverts the result, making the cells white (foreground).

3.  **Cleaning (Morphology):**

      * **`cv2.morphologyEx(..., cv2.MORPH_OPEN, ...)`** is performed. This is an **Erosion followed by a Dilation**. It's highly effective at removing small white specks (background noise) that were incorrectly thresholded.
      * **`cv2.morphologyEx(..., cv2.MORPH_CLOSE, ...)`** is performed. This is a **Dilation followed by an Erosion**. It's effective at filling small black holes *inside* the white yeast objects.

### 3\. Feature Extraction (Density Analysis)

The ANN cannot understand images directly; it needs numbers. This step calculates image-wide features from the cleaned segmentation mask.

1.  **Find Contours:** `cv2.findContours` is called on the final binary mask to get a list of all detected objects.
2.  **Filter Contours:** The list is filtered to remove tiny contours smaller than `MIN_CONTOUR_AREA`.
3.  **Calculate Features (`extractDensityFeatures`):**
      * `contour_count`: The total number of valid contours (yeast objects) found.
      * `total_yeast_area`: The sum of the pixel areas of all valid contours.
      * `image_area_pixels`: The total number of non-black pixels in the cropped ROI (this is our denominator).
      * **`area_fraction` (Key Feature):** `total_yeast_area / image_area_pixels`. This is the core measure of density.
      * `average_area`: `total_yeast_area / contour_count`. This gives a sense of the average size of the detected objects.

### 4\. ANN Training & Evaluation

The calculated features for each image are used to train the classifier.

1.  **Data Prep:** All features are collected into a `pandas` DataFrame, with a 'label' column (0 for SC, 1 for SP).
2.  **Train/Test Split:** The data is split into a training set (70%) and a testing set (30%) using `train_test_split`.
3.  **Scaling:** The features are scaled using `StandardScaler`. This is **essential** for ANNs, as it ensures all features have a similar range and contribute equally.
4.  **Training:** An `MLPClassifier` (Multi-Layer Perceptron) is created. The model is trained using the `.fit()` method on the scaled training data.
5.  **Evaluation:** The trained model makes predictions on the unseen test data. A **Classification Report** and **Confusion Matrix** are printed to show the model's accuracy, precision, and recall, giving a clear picture of its performance.

-----

## Visual Output 📊

When the script is in classification mode (after training), it will display a plot for each image you provide:

  * **Plot 1: Original Image:** The full, untouched image you provided.
  * **Plot 2: Segmentation Mask:** The final cleaned black and white mask showing what the script identified as yeast.
  * **Plot 3: Detected Contours:** The cropped image with green outlines drawn around all the yeast objects it counted.
  * **Text Below Plot:** A summary of the extracted features (`Count`, `AreaFrac`, `AvgArea`) and the final **ANN Prediction** (e.g., "S. Cerevisiae (SC - Low Density)").

-----

## Installation 🚀

1.  **Prerequisites:**

      * Python 3.7+
      * `pip` or `conda` for package management

2.  **Clone the Repository (Optional):**

    ```bash
    git clone https://your-repo-url/Yeast-Classifier.git
    cd Yeast-Classifier
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:

    ```
    opencv-python-headless
    numpy
    matplotlib
    scikit-learn
    pandas
    ```

    Then, install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

-----

## Usage ▶️

### Step 1: Set Up Your Data

Before running, you **must** update the `ROOT_DIR_SC` and `ROOT_DIR_SP` variables in the script:

```python
# --- 1. SET YOUR FOLDER AND FILE TYPES ---
# Assign labels based on USER DENSITY RULE: SC=Low Density(0), SP=High Density(1)
ROOT_DIR_SC = r"C:\path\to\your\Saccharomyces_Cerevisiae_images" # Label 0 (Low Density)
ROOT_DIR_SP = r"C:\path\to\your\Saccharomyces_Pastorianus_images" # Label 1 (High Density)
```

  * Place your **low-density** yeast images in the `ROOT_DIR_SC` folder.
  * Place your **high-density** yeast images in the `ROOT_DIR_SP` folder.

### Step 2: Run the Script

1.  Open a terminal or command prompt.

2.  Navigate to the directory containing the script.

3.  Run the script:

    ```bash
    python your_script_name.py
    ```

4.  **Training:** The script will first process all images in your `ROOT_DIR` folders, extract features, and train the ANN. It will print the evaluation report to the console.

5.  **Classification:** After training, the script will enter a loop and prompt you:

    ```
    --- Ready to Classify New Images by Density ---
    Enter the full path to a new yeast image (or type 'quit' to exit):
    ```

6.  Paste the full file path to any yeast image (it doesn't have to be in the training folders) and press **Enter**.

7.  A Matplotlib window will pop up showing the classification results. Close the window to classify another image.

8.  Type `quit` and press **Enter** to exit the program.

-----

## Tunable Parameters Explained 🔧

You can change these values at the top of the script (Section 2) to improve segmentation performance on your specific images.

### Registration Parameters

  * `ZOOM_FACTOR`: Controls the crop. `0.8` crops to a circle 80% the size of the original ROI.

### Segmentation Parameters

  * `MEDIAN_BLUR_KERNEL`: Size of the smoothing filter. **Increase (e.g., 5, 7)** if your image has a lot of "salt-and-pepper" noise.
  * `CLAHE_CLIP_LIMIT`: How much to limit contrast. **Increase (e.g., 4.0, 5.0)** if cells are very faint; **Decrease (e.g., 2.0)** if it's over-enhancing background noise.
  * `ADAPTIVE_THRESH_BLOCK_SIZE`: **Crucial.** Size of the neighborhood for thresholding. Must be odd. **Increase (e.g., 35, 45)** if the background lighting varies slowly; **Decrease (e.g., 11, 15)** for more local detail.
  * `ADAPTIVE_THRESH_C`: **Crucial.** A constant subtracted from the local mean. This fine-tunes the sensitivity. **Increase (e.g., 7, 10)** if you are picking up too much background (makes it harder to be "foreground"); **Decrease (e.g., 2, 3)** if you are missing faint cells (makes it easier to be "foreground").
  * `MORPH_KERNEL_SIZE_OPEN`: Size of the kernel to remove noise. **Increase (e.g., 5, 7)** to remove larger background spots.
  * `MORPH_KERNEL_SIZE_CLOSE`: Size of the kernel to fill holes. **Increase (e.g., 5, 7)** to fill larger gaps inside cells or connect broken parts.
  * `MORPH_OPEN_ITERATIONS`: How many times to apply opening. **Increase (e.g., 2)** to be more aggressive in removing noise.
  * `MIN_CONTOUR_AREA`: The final filter. **Increase (e.g., 50, 100)** if small noise spots are still being counted as cells.

-----

## Future Improvements 💡

  * **Hyperparameter Tuning:** The ANN architecture (`hidden_layer_sizes=(16, 8)`) is simple. A more systematic search (like `GridSearchCV`) could find a better-performing model.
  * **Feature Engineering:** The model *only* uses density. It could be improved by adding **shape features** (like average circularity, aspect ratio, etc.) to the feature vector.
  * **Robustness:** The density hypothesis is strong. If `SC` sometimes appears dense or `SP` sparse, the model will fail. A model trained on both density *and* shape might be more robust.
  * **Deep Learning (CNN):** A Convolutional Neural Network (CNN) could be trained on the cropped images directly, learning features automatically, which might yield higher accuracy if enough training data is available.
