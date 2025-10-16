import cv2
import numpy as np
import glob
import os

# Define the directories for predictions and ground truths
preds_dir = 'data/output/depths/'
gts_dir = 'data/synthetic_data/clean_512_label/'

# Get the list of all predicted depth maps
pred_paths = sorted(glob.glob(os.path.join(preds_dir, '*.png')))

mse_list = []

# --- Main Change: Iterate through predictions and find matching ground truth ---
print(f"Found {len(pred_paths)} prediction files in '{preds_dir}'")

for pred_path in pred_paths:
    # 1. Get the base filename (e.g., 'image_001.png') from the prediction path
    base_filename = os.path.basename(pred_path)
    
    # 2. Construct the full path for the corresponding ground truth file
    gt_path = os.path.join(gts_dir, base_filename)
    
    print(f"\nProcessing: {base_filename}")
    
    # 3. Check if the corresponding ground truth file actually exists
    if not os.path.exists(gt_path):
        print(f"--> SKIPPING: Ground truth not found at '{gt_path}'")
        continue

    # Load prediction and ground truth images
    pred = cv2.imread(pred_path)
    gt = cv2.imread(gt_path)

    # Check if images were loaded successfully
    if pred is None:
        print(f"--> SKIPPING: Unable to read prediction image '{pred_path}'")
        continue
    if gt is None:
        print(f"--> SKIPPING: Unable to read ground truth image '{gt_path}'")
        continue

    # Normalize pixel values to the range [0, 1]
    pred = pred.astype(np.float32) / 255.0
    gt = gt.astype(np.float32) / 255.0

    # Check for shape mismatch before calculating MSE
    if pred.shape != gt.shape:
        print(f"--> SKIPPING: Shape mismatch between pred {pred.shape} and gt {gt.shape}")
        continue

    # Calculate and store the Mean Squared Error for the pair
    mse = np.mean((pred - gt) ** 2)
    mse_list.append(mse)
    print(f"--> SUCCESS: MSE = {mse:.6f}")


# --- Final Calculation ---
print("\n" + "="*40)
if mse_list:
    average_mse = np.mean(mse_list)
    print(f"✅ Final Average MSE across {len(mse_list)} valid pairs: {average_mse:.6f}")
else:
    print("❌ Average MSE: nan (no valid image pairs were found/processed)")
print("="*40)