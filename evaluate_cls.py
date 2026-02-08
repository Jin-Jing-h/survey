import argparse
import os
import torch
import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
import sys
from os import path as osp

# Add current directory to path
sys.path.append(osp.join(os.getcwd()))

from utils.options import dict2str, parse
from utils import get_env_info, set_random_seed
from data import create_dataset, create_dataloader
from models import create_model

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained .pth model. Overrides yaml.')
    parser.add_argument('--val_set', type=str, default='val_raindrop', help='Name of the validation set in yaml to use.')
    args = parser.parse_args()
    
    # Parse YAML
    opt = parse(args.opt, is_train=False)
    
    # Override model path if provided in CLI
    if args.model_path is not None:
        opt['path']['pretrain_network_g'] = args.model_path
        opt['path']['strict_load_g'] = True # Usually strictly load for evaluation
        print(f"Overriding model path to: {args.model_path}")

    # Force distributed to False for simple evaluation
    opt['dist'] = False
    opt['num_gpu'] = 1
    opt['rank'] = 0
    opt['world_size'] = 1
    
    return opt, args

def main():
    # 1. Initialization
    opt, args = parse_options()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 2. Create Dataset and Dataloader
    val_opt = opt['datasets'].get(args.val_set)
    if not val_opt:
        raise ValueError(f"Validation set '{args.val_set}' not found in YAML datasets.")
    
    # Ensure dataset returns labels/levels
    # (Assuming your Dataset implementation reads level_json if provided)
    if 'level_json' not in val_opt and 'train' in opt['datasets']:
        # Fallback: try to borrow level_json from train config if missing in val
        if 'level_json' in opt['datasets']['train']:
             val_opt['level_json'] = opt['datasets']['train']['level_json']
             print(f"Note: Borrowed level_json from train config: {val_opt['level_json']}")

    val_set = create_dataset(val_opt)
    val_loader = create_dataloader(
        val_set,
        val_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed']
    )
    print(f"Dataset Loaded. Total samples: {len(val_set)}")

    # 3. Create Model
    model = create_model(opt)
    model.net_g.eval()
    
    # Check if logic for 'forward_with_pad_and_cls' exists (from your uploaded code)
    if not hasattr(model, 'forward_with_pad_and_cls'):
        raise AttributeError("Your ImageCleanModel code must have 'forward_with_pad_and_cls' defined as per your uploaded files.")

    # 4. Inference Loop
    pred_list = []
    gt_list = []
    
    window_size = opt['val'].get('window_size', 8)
    print(f"Starting Inference with window_size={window_size}...")

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Move data to device
            model.feed_data(data)
            
            # Check if GT Level exists
            if model.level is None:
                print(f"[Warning] Sample {i} has no GT level. Skipping metrics for this sample.")
                continue

            # Forward pass to get Logits
            # Note: We don't care about the restored image 'output' here, only 'logits'
            _, logits = model.forward_with_pad_and_cls(model.lq, window_size=window_size)
            
            if logits is None:
                print(f"[Error] Model returned None for logits at sample {i}.")
                continue

            # --- Ordinal Regression Decoding Logic ---
            # Based on your training code: pred = sum(sigmoid(logits) > 0.5)
            # Logits shape: [B, 3] corresponding to thresholds >=1, >=2, >=3
            probs = torch.sigmoid(logits)
            pred_level = (probs > 0.5).long().sum(dim=1) # Result is 0, 1, 2, or 3
            
            gt_level = model.level.view(-1)

            # Store results
            pred_list.extend(pred_level.cpu().numpy().tolist())
            gt_list.extend(gt_level.cpu().numpy().tolist())

            if i % 100 == 0:
                print(f"Processed {i}/{len(val_loader)} batches...")

    # 5. Calculate Metrics
    if len(gt_list) == 0:
        print("No valid Ground Truth levels found. Check your dataset configuration.")
        return

    # Convert to numpy
    y_true = np.array(gt_list)
    y_pred = np.array(pred_list)
    
    # 5a. Top-1 Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # 5b. MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 5c. Confusion Matrix
    # Labels should be 0, 1, 2, 3
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 5d. Per-Class Accuracy
    # Diagonal elements / sum of row (support)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Handle NaN if a class is missing in GT
    per_class_acc = np.nan_to_num(per_class_acc)

    # 6. Print Report
    print("\n" + "="*50)
    print("CLASSIFICATION EVALUATION RESULTS")
    print("="*50)
    print(f"Total Samples Evaluated: {len(y_true)}")
    print("-" * 30)
    print(f"Top-1 Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
    print(f"Mean Absolute Error:  {mae:.4f}")
    print("-" * 30)
    print("Per-Class Accuracy:")
    for idx, cls_acc in enumerate(per_class_acc):
        count = cm[idx].sum()
        print(f"  Level {idx}: {cls_acc:.4f} ({cls_acc*100:.2f}%) - Support: {count}")
    
    print("-" * 30)
    print("Confusion Matrix (Rows=GT, Cols=Pred):")
    print("      Pred 0  Pred 1  Pred 2  Pred 3")
    for idx, row in enumerate(cm):
        print(f"GT {idx}  {row}")
    print("="*50)

if __name__ == '__main__':
    main()