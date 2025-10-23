# --- 0. LIBRARY ---
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. FOLDERS---
ROOT_DIR_SC = r"C:\Users\Thanakorn\University\Image Processing\Final Project\Saccharomyces_Cerevisiae"
ROOT_DIR_SP = r"C:\Users\Thanakorn\University\Image Processing\Final Project\Saccharomyces_Pastorianus"
IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'tif', 'tiff']

# --- 2. TUNEABLE PARAMETERS ---
ZOOM_FACTOR = 0.7
MEDIAN_BLUR_KERNEL = 3
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)
ADAPTIVE_THRESH_BLOCK_SIZE = 25
ADAPTIVE_THRESH_C = 5
MORPH_KERNEL_SIZE_OPEN = 3
MORPH_KERNEL_SIZE_CLOSE = 5
MORPH_OPEN_ITERATIONS = 1
MIN_CONTOUR_AREA = 50

# --- Helper Functions ---
def findImagePaths(root_dir, extensions):
    print(f"Searching for images in: {root_dir}")
    image_files = []
    for ext in extensions:
        search_pattern = os.path.join(root_dir, '**', f'*.{ext}')
        image_files.extend(glob.glob(search_pattern, recursive=True))
    print(f" -> Found {len(image_files)} images.")
    return image_files

def findRoiCircle(img_gray):
    blurred = cv.GaussianBlur(img_gray, (35, 35), 0)
    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv.contourArea)
    if cv.contourArea(largest_contour) < 0.1 * img_gray.size: return None
    ((center_x, center_y), radius) = cv.minEnclosingCircle(largest_contour)
    if radius < 50 or radius > max(img_gray.shape): return None
    return (int(center_x), int(center_y), int(radius))

def centerAndCropImage(img_color, circle_params, zoom=1.0):
    if circle_params is None: return None
    (center_x, center_y, radius) = circle_params
    img_height, img_width = img_color.shape[:2]
    desired_center_x, desired_center_y = img_width // 2, img_height // 2
    shift_x, shift_y = desired_center_x - center_x, desired_center_y - center_y
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv.warpAffine(img_color, M, (img_width, img_height), borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))

    effective_radius = int(radius * zoom)
    
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv.circle(mask, (desired_center_x, desired_center_y), radius, 255, -1)
    masked_img = cv.bitwise_and(shifted_img, shifted_img, mask=mask)

    y_min, y_max = max(0, desired_center_y - effective_radius), min(img_height, desired_center_y + effective_radius)
    x_min, x_max = max(0, desired_center_x - effective_radius), min(img_width, desired_center_x + effective_radius)
    final_cropped_img = masked_img[y_min:y_max, x_min:x_max]
    if final_cropped_img.shape[0] == 0 or final_cropped_img.shape[1] == 0: return None

    return final_cropped_img, effective_radius

def segmentYeast(img_color_cropped, clahe_obj):
    if img_color_cropped is None or img_color_cropped.size == 0: return None, None, [], 0
    img_gray = cv.cvtColor(img_color_cropped, cv.COLOR_BGR2GRAY)
    denoised_gray = cv.medianBlur(img_gray, MEDIAN_BLUR_KERNEL)
    enhanced_gray = clahe_obj.apply(denoised_gray)
    binary_mask = cv.adaptiveThreshold(
        enhanced_gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        ADAPTIVE_THRESH_BLOCK_SIZE,
        ADAPTIVE_THRESH_C
        )
    if cv.countNonZero(binary_mask) < MIN_CONTOUR_AREA:
        ret_otsu, binary_mask = cv.threshold(enhanced_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel_open = np.ones((MORPH_KERNEL_SIZE_OPEN, MORPH_KERNEL_SIZE_OPEN), np.uint8)
    kernel_close = np.ones((MORPH_KERNEL_SIZE_CLOSE, MORPH_KERNEL_SIZE_CLOSE), np.uint8)
    mask_opened = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel_open, iterations=MORPH_OPEN_ITERATIONS)
    final_mask = cv.morphologyEx(mask_opened, cv.MORPH_CLOSE, kernel_close, iterations=1)

    contours, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv.contourArea(c) > MIN_CONTOUR_AREA]
    contour_count = len(filtered_contours)

    img_with_contours = img_color_cropped.copy()
    cv.drawContours(img_with_contours, filtered_contours, -1, (0, 255, 0), 1)

    return final_mask, img_with_contours, filtered_contours, contour_count

def extractDensityFeatures(contours, image_area_pixels):
    contour_count = len(contours)
    total_yeast_area = sum(cv.contourArea(c) for c in contours)

    # Avoid division by zero
    if image_area_pixels > 0:
        area_fraction = total_yeast_area / image_area_pixels
    else:
        area_fraction = 0

    if contour_count > 0:
        average_area = total_yeast_area / contour_count
    else:
        average_area = 0

    features = {
        'contour_count': contour_count,
        'total_yeast_area': total_yeast_area,
        'area_fraction': area_fraction,
        'average_area': average_area
    }
    return features

def classifyNewImageDensity(image_path, mlp_model, scaler_obj, clahe_obj):
    base_name = os.path.basename(image_path)
    print(f"\n--- Classifying New Image: {base_name} ---")
    img_color = cv.imread(str(image_path))
    if img_color is None: print("Error: Could not read image."); return

    try:
        height, width, _ = img_color.shape
        img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
        img_gray_denoised_roi = cv.medianBlur(img_gray, 3)
        enhanced_gray_roi = clahe_obj.apply(img_gray_denoised_roi)
        circle_params = findRoiCircle(enhanced_gray_roi)
        if circle_params is None: print("Warning: No ROI circle found."); return

        final_cropped_image, effective_radius = centerAndCropImage(img_color, circle_params, zoom=ZOOM_FACTOR)
        if final_cropped_image is None: print("Error: Cropping failed."); return

        img_gray_cropped = cv.cvtColor(final_cropped_image, cv.COLOR_BGR2GRAY)
        image_area_pixels = cv.countNonZero(img_gray_cropped)

        binary_mask, img_contours_vis, contours, count = segmentYeast(final_cropped_image, clahe_obj)
        if binary_mask is None: print("Segmentation failed."); return

        if not contours:
             print("Warning: No contours found after segmentation. Cannot classify.")
             features = {'contour_count': 0, 'total_yeast_area': 0, 'area_fraction': 0, 'average_area': 0}
        else:
            features = extractDensityFeatures(contours, image_area_pixels)

        print(" -> Calculated Features:", features)

        df_new = pd.DataFrame([features])

        try: df_new = df_new[scaler_obj.feature_names_in_]
        except AttributeError: pass # Assume order correct if names not stored
        new_features_scaled = scaler_obj.transform(df_new)
        prediction = mlp_model.predict(new_features_scaled)[0]

        # Labels: SC=0 (Low Density), SP=1 (High Density)
        if prediction == 0:
            predicted_class_name = "S. Cerevisiae"
        else:
            predicted_class_name = "S. Pastorianus"
        print(f" -> ANN Prediction: {predicted_class_name} (Class {prediction})")
        
        pixel_size = 200 * (10**-6) / width
        
        area_square = pixel_size * pixel_size * features['average_area'] * 1e12

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f'Classification for: {base_name}\nPrediction: {predicted_class_name}', fontsize=16)

        axes[0].imshow(cv.cvtColor(img_color, cv.COLOR_BGR2RGB))
        axes[0].set_title('1. Original Image')
        axes[0].axis('off')

        axes[1].imshow(cv.cvtColor(img_contours_vis, cv.COLOR_BGR2RGB))
        axes[1].set_title(f'2. Detected Contours ({count})')
        axes[1].axis('off')

        plt.figtext(0.5,
                    0.02,
                    f"Features: Count={features['contour_count']}, AreaFrac={features['area_fraction']:.3f}, AvgArea={features['average_area']:.1f} ActualSize={area_square:.3f} μm²",
                    ha="center",
                    fontsize=10,
                    bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

        plt.tight_layout(rect=[0, 0.08, 1, 0.93])
        plt.show()

    except Exception as e:
        print(f"An error occurred during classification: {e}")

def main():
    
    image_paths_sc = findImagePaths(ROOT_DIR_SC, IMAGE_EXTENSIONS) # 0
    image_paths_sp = findImagePaths(ROOT_DIR_SP, IMAGE_EXTENSIONS) # 1
    all_image_paths = [(path, 0) for path in image_paths_sc] + \
                      [(path, 1) for path in image_paths_sp]
    print(f"\nTotal images found for training/testing: {len(all_image_paths)}\n")
    if not all_image_paths: return

    clahe_obj = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    all_features_data = []
    print("Starting Feature Extraction for Training Data ...")

    for image_path, label in all_image_paths:
        base_name = os.path.basename(image_path)
        try:
            img_color = cv.imread(str(image_path))
            if img_color is None: continue

            img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
            img_gray_denoised_roi = cv.medianBlur(img_gray, 3)
            enhanced_gray_roi = clahe_obj.apply(img_gray_denoised_roi)
            circle_params = findRoiCircle(enhanced_gray_roi)
            if circle_params is None: continue

            final_cropped_image, effective_radius = centerAndCropImage(img_color, circle_params, zoom=ZOOM_FACTOR)
            if final_cropped_image is None: continue

            img_gray_cropped = cv.cvtColor(final_cropped_image, cv.COLOR_BGR2GRAY)
            image_area_pixels = cv.countNonZero(img_gray_cropped)
            if image_area_pixels == 0: continue

            _, _, contours, _ = segmentYeast(final_cropped_image, clahe_obj)

            image_features = extractDensityFeatures(contours, image_area_pixels)
            image_features['label'] = label
            all_features_data.append(image_features)

        except Exception as e: print(f"Error processing {base_name} for training: {e}")
    print("\nTraining Feature Extraction Complete.")

    if not all_features_data: print("No features extracted."); return
    df = pd.DataFrame(all_features_data)
    X = df.drop('label', axis=1); y = df['label']
    print(f"\nTotal images for training/testing: {len(df)}")
    print("Class Distribution:\n", y.value_counts())
    if len(np.unique(y)) < 2: print("\nError: Need data from both classes."); return

    # --- ANN Training ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nTraining MLP Classifier on Density Features...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=1000,
        activation='relu',
        solver='adam',
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50, # More patience
        verbose=False
        )
    mlp.fit(X_train_scaled, y_train)
    print("Training complete.")

    print("\n--- Evaluating Model on Test Set ---")
    y_pred = mlp.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['S. cerevisiae (SC - 0)', 'S. pastorianus (SP - 1)']))

    print("\n--- Ready to Classify New Images ---")
    while True:
        try:
            new_image_path = input("Enter the full path to a new yeast image (or type 'quit' to exit): ").strip().strip('"')
            if new_image_path.lower() == 'quit': break
            if not os.path.exists(new_image_path): print("Error: File not found."); continue
            classifyNewImageDensity(new_image_path, mlp, scaler, clahe_obj)
        except Exception as e: print(f"An error occurred: {e}")
    print("\nExiting program.")

if __name__ == "__main__":
    main()