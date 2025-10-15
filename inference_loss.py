import torch
import torch.nn.functional as F
import cv2
import numpy as np
import glob
import os

preds = sorted(glob.glob('data/output/depths/*.png'))
gts = sorted(glob.glob('data/synthetic_data/clean_512_label/*.png'))

mse_list = []

for pred_path, gt_path in zip(preds, gts):
    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
        continue

    pred = cv2.imread(pred_path)
    gt = cv2.imread(gt_path)

    if pred is None or gt is None:
        print(f"Skipping {pred_path} or {gt_path}: unable to read image")
        continue

    pred = pred.astype(np.float32) / 255.0
    gt = gt.astype(np.float32) / 255.0

    if pred.shape != gt.shape:
        print(f"Skipping {pred_path} / {gt_path}: shape mismatch {pred.shape} vs {gt.shape}")
        continue

    mse = np.mean((pred - gt) ** 2)
    mse_list.append(mse)

if mse_list:
    print("Average MSE:", np.mean(mse_list))
else:
    print("Average MSE: nan (no valid image pairs found)")
