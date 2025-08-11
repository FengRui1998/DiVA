import sys
import os

# Add parent directory to sys.path to allow importing from 'utils' and other sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add 'data_loader' directory to sys.path, assuming it's a sibling or specified path
sys.path.append(os.path.abspath('data_loader'))  # This path might need adjustment based on project structure

# Standard library imports
import argparse
import math
import time
import datetime
import warnings

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for Matplotlib, suitable for servers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tqdm  # For progress bars

# Local application/library specific imports
from utils import (  # Assuming these are from your utils.py
    RecorderMeter, AverageMeter, ProgressMeter,
    save_checkpoint, plot_confusion_matrix, plot_roc_curve,
    set_random_seed, FocalLossWithLogits, plot_fold_ROC_CURVE  # Added plot_fold_ROC_CURVE assuming it's in utils
)
from dataset_loader import get_data_loaders  # From the data_loader directory
from diva.diva_model import MISA_FR, Share_Discriminator, Private_Discriminator, \
    GradientReversalLayer  # Model components

warnings.filterwarnings("ignore", category=UserWarning)  # Suppress specific UserWarnings


# ==============================================================================
# Argument Parsing
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DiVA Model Training and Evaluation")
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed to use for reproducibility.')  # Default was string "1"
    parser.add_argument('--gpu', type=str, default="0", help='GPU IDs to use (e.g., "0" or "0,1").')  # Default was "4"
    parser.add_argument('--model', type=str, default='DiVA', help='Model name.')
    parser.add_argument('--log', type=str, default='Loss', help='Prefix for log directory name.')
    parser.add_argument('--sim_loss', type=str, default='MMDLoss', choices=['MMDLoss', 'KernelCenteredDistance'],
                        help='Similarity loss type.')
    parser.add_argument('--diff_loss', type=str, default='HSICLoss', choices=['HSICLoss', 'MSE'],
                        help='Difference loss type.')
    parser.add_argument('--dataset', type=str, default='Combine', choices=['Combine', 'ChangZhou', 'NanJing', 'SuZhou'],
                        help="Dataset name/location identifier.")
    parser.add_argument('--data_loader', type=str, default='Clip', choices=['Clip', 'Full'],
                        help='Dataset loading mode: Clip or Full.')
    parser.add_argument('--use_augmentation', type=bool, default=True, help='Use data augmentation during training.')
    parser.add_argument('--criterion', type=str, default='BCE', choices=['BCE', 'Focal'],
                        help='Classification loss criterion.')
    parser.add_argument('--drop_last', type=bool, default=False, help='Whether to drop the last incomplete batch.')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of total epochs to run.')
    parser.add_argument('--batch-size', type=int, default=8, help='Mini-batch size.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer.')
    parser.add_argument('--print-freq', type=int, default=5, help='Frequency of printing training progress.')
    parser.add_argument('--frame_num', type=int, default=32, help='Number of frames per video clip.')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size (height and width).')

    # Loss weights
    parser.add_argument('--lambda-sim', type=float, default=0.0001, help='Weight for similarity loss (con_loss).')
    parser.add_argument('--lambda-diff', type=float, default=1.0,
                        help='Weight for difference loss (dif_loss).')  # Default was 1
    parser.add_argument('--lambda-adv', type=float, default=1.0, help='Weight for adversarial losses.')  # Default was 1

    args = parser.parse_args()
    # Set visible GPUs using environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args


# ==============================================================================
# Main Training and Validation Function per Fold
# ==============================================================================
def main(current_fold_idx, args):  # Renamed 'set' to 'current_fold_idx' for clarity
    data_set_identifier = current_fold_idx + 1  # For logging purposes, 1-indexed

    # Setup log paths
    log_dir_base = log_file  # Global variable 'log_file' defined in __main__
    current_fold_log_path = os.path.join(log_dir_base, f'{args.dataset}-set{data_set_identifier}-log/')
    log_txt_path = os.path.join(current_fold_log_path, 'log.txt')
    log_loss_path = os.path.join(current_fold_log_path, 'loss.png')
    log_acc_path = os.path.join(current_fold_log_path, 'acc.png')
    log_uar_path = os.path.join(current_fold_log_path, 'uar.png')
    log_confusion_matrix_path = os.path.join(current_fold_log_path, 'cn.png')
    log_roc_curve_path = os.path.join(current_fold_log_path, 'roc_curve.png')
    checkpoint_path = current_fold_log_path  # Directory to save checkpoints
    os.makedirs(current_fold_log_path, exist_ok=True)

    best_uar = 0.0
    best_acc = 0.0
    recorder = RecorderMeter(args.epochs)  # Utility to record and plot metrics over epochs

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MISA_FR(args=args).to(device)
    model_D_S = Share_Discriminator(input_dim=256).to(device)  # Assuming represent_dim is 256
    model_D_P = Private_Discriminator(input_dim=256).to(device)  # Assuming represent_dim is 256

    # Log arguments
    with open(log_txt_path, 'a') as f:
        f.write("Command line arguments:\n")
        for k, v in vars(args).items():
            f.write(f"{k} = {v}\n")
        f.write("\n")

    # Define loss function (criterion) for classification
    if args.criterion == 'BCE':
        criterion_cls = nn.BCEWithLogitsLoss().to(device)
    elif args.criterion == 'Focal':
        criterion_cls = FocalLossWithLogits(alpha=0.25, gamma=2, reduction='mean').to(device)
        print(f"Using FocalLoss for fold {data_set_identifier}")

    # Define optimizers and schedulers
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    optimizer_D_S = torch.optim.AdamW(params=model_D_S.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_D_S = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D_S, T_max=args.epochs)

    optimizer_D_P = torch.optim.AdamW(params=model_D_P.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_D_P = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D_P, T_max=args.epochs)

    # Data loading
    train_loader, val_loader = get_data_loaders(args, data_set_identifier)  # Pass 1-indexed identifier

    # Start training loop
    for epoch in range(args.epochs):  # 0 to epochs-1
        current_epoch_display = epoch + 1  # 1-indexed for display
        epoch_info_header = f'******************** Epoch {current_epoch_display}/{args.epochs} ********************'
        start_time_epoch = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(epoch_info_header + '\n')
            print(epoch_info_header)
            f.write(f'Current learning rate: {current_lr}\n')
            print(f'Current learning rate: {current_lr}')

        # Train for one epoch
        train_loss, train_acc, train_uar, train_war, train_f1 = train(
            train_loader, model, model_D_S, model_D_P, criterion_cls,
            optimizer, optimizer_D_S, optimizer_D_P,
            current_epoch_display, args, log_txt_path, device
        )
        scheduler.step()
        scheduler_D_S.step()
        scheduler_D_P.step()

        # Evaluate on validation set
        val_loss, val_acc, val_uar, val_war, val_f1 = validate(
            val_loader, model, criterion_cls,  # Discriminators not used directly in validation loss for main model
            current_epoch_display, args, log_txt_path, device
        )

        # Determine if current model is the best based on UAR and ACC
        is_best = False
        if val_uar > best_uar:
            is_best = True
            best_uar = val_uar
            best_acc = val_acc
        elif val_uar == best_uar and val_acc > best_acc:  # If UAR is same, pick based on higher ACC
            is_best = True
            best_acc = val_acc

        save_checkpoint({
            'epoch': current_epoch_display,
            'state_dict': model.state_dict(),
            'best_uar': best_uar,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_path)  # Saves 'checkpoint.pth' and 'best_model.pth' if is_best

        # Print and save log
        epoch_duration = time.time() - start_time_epoch
        recorder.update(epoch, train_loss, val_loss, train_acc, val_acc, train_uar, val_uar)
        recorder.plot_loss_curve(log_loss_path)
        recorder.plot_accuracy_curve(log_acc_path)
        recorder.plot_uar_curve(log_uar_path)

        summary_msg = (
            f'Epoch {current_epoch_display} Summary:\n'
            f'  Best Validation ACC: {best_acc:.4f}\n'
            f'  Best Validation UAR: {best_uar:.4f}\n'
            f'  Epoch Time: {epoch_duration:.2f}s'
        )
        print(summary_msg)
        with open(log_txt_path, 'a') as f:
            f.write(summary_msg + '\n\n')

    # After all epochs, compute final scores using the best model on the validation set
    final_acc, final_uar, final_war, final_f1, final_roc_auc, fpr, tpr = compute_score(
        val_loader, model, checkpoint_path,
        log_confusion_matrix_path, log_roc_curve_path,
        log_txt_path, data_set_identifier, args.class_names, device
    )
    return final_acc, final_uar, final_war, final_f1, final_roc_auc, fpr, tpr


# ==============================================================================
# Training Function
# ==============================================================================
def train(dataloader, model, model_D_S, model_D_P, criterion_cls,
          optimizer_main, optimizer_D_S, optimizer_D_P,  # Renamed optimizer to optimizer_main
          epoch, args, log_txt_path, device):
    losses_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('ACC', ':.4f')
    uar_meter = AverageMeter('UAR', ':.4f')
    war_meter = AverageMeter('WAR', ':.4f')
    f1_meter = AverageMeter('F1', ':.4f')
    progress = ProgressMeter(len(dataloader),
                             [losses_meter, acc_meter, uar_meter, war_meter, f1_meter],
                             prefix=f"Epoch: [{epoch}] Train",
                             log_txt_path=log_txt_path)

    model.train()
    model_D_S.train()
    model_D_P.train()

    running_loss_epoch = 0.0
    all_labels_epoch = []
    all_predictions_epoch = []

    for i, (video, audio, labels) in enumerate(dataloader):
        video = video.to(device)
        audio = audio.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # For BCEWithLogitsLoss

        # --- Train Main Model (Encoders and Classifier) ---
        optimizer_main.zero_grad()

        # Forward pass through the main model
        output_logits, shared_v, private_v, shared_a, private_a, consistency_loss, difference_loss = model(video, audio)

        # 1. Classification loss
        loss_cls = criterion_cls(output_logits, labels)

        # 2. Shared feature adversarial loss (to make shared features modality-invariant)
        # Apply GRL to shared features before feeding to Share_Discriminator
        shared_v_grl = GradientReversalLayer.apply(shared_v)
        shared_a_grl = GradientReversalLayer.apply(shared_a)
        # Discriminator predicts modality based on shared features
        # Note: Discriminator concatenates shared_v_grl and shared_a_grl internally if that's its design
        # The labels here are for the *main model* to *fool* the discriminator.
        # So, if D_S is trained to output 0 for video and 1 for audio,
        # the main model wants D_S(shared_v_grl) to be 1 and D_S(shared_a_grl) to be 0.
        # Or, more commonly, D_S output fixed labels and main model tries to make D_S output the *opposite*.
        # The current implementation of Share_Discriminator expects concatenated input.
        # The labels indicate the true origin (0 for video, 1 for audio).
        # The main model's adversarial loss aims to make the discriminator *wrong*.
        # F.cross_entropy expects logits and class indices.
        shared_logits_adv = model_D_S(shared_v_grl, shared_a_grl)  # This call assumes D_S handles concatenation
        # Modality labels: 0 for video features, 1 for audio features
        modality_labels_shared = torch.cat([
            torch.zeros(shared_v.size(0), device=device),
            torch.ones(shared_a.size(0), device=device)
        ]).long()
        loss_adv_shared = F.cross_entropy(shared_logits_adv, modality_labels_shared)

        # 3. Private feature discriminative loss (to make private features modality-specific for D_P)
        # This loss is for training D_P, not directly for the main model here unless GRL is applied.
        # The original code includes it in the main loss, implying GRL for private features too if adversarial.
        # Based on the formula: loss = loss_cls + ... + args.lambda_adv * (shared_loss + private_loss)/2
        # It seems private_loss is also an adversarial component for the main model.
        private_v_grl = GradientReversalLayer.apply(private_v)  # Assuming GRL for private for main model training
        private_a_grl = GradientReversalLayer.apply(private_a)
        private_logits_adv = model_D_P(private_v_grl, private_a_grl)
        modality_labels_private = torch.cat([
            torch.zeros(private_v.size(0), device=device),
            torch.ones(private_a.size(0), device=device)
        ]).long()
        loss_adv_private = F.cross_entropy(private_logits_adv, modality_labels_private)

        # Total loss for the main model
        total_loss_main = (loss_cls +
                           args.lambda_sim * consistency_loss +
                           args.lambda_diff * difference_loss +
                           args.lambda_adv * (loss_adv_shared + loss_adv_private) / 2.0)

        total_loss_main.backward()
        optimizer_main.step()

        # --- Train Discriminators (D_S and D_P) ---
        # 4. Update Share_Discriminator (D_S)
        optimizer_D_S.zero_grad()
        # Detach shared features so gradients don't flow back to encoders during D_S update
        shared_logits_D_S = model_D_S(shared_v.detach(), shared_a.detach())
        loss_D_S_train = F.cross_entropy(shared_logits_D_S, modality_labels_shared)  # D_S tries to correctly classify
        loss_D_S_train.backward()
        optimizer_D_S.step()

        # 5. Update Private_Discriminator (D_P)
        optimizer_D_P.zero_grad()
        # Detach private features
        private_logits_D_P = model_D_P(private_v.detach(), private_a.detach())
        loss_D_P_train = F.cross_entropy(private_logits_D_P, modality_labels_private)  # D_P tries to correctly classify
        loss_D_P_train.backward()
        optimizer_D_P.step()

        # Calculate predictions and metrics for the current batch from main model's output
        predictions_batch = (torch.sigmoid(output_logits.detach()) > 0.5).float()

        batch_acc = accuracy_score(labels.cpu().numpy(), predictions_batch.cpu().numpy())
        batch_uar = recall_score(labels.cpu().numpy(), predictions_batch.cpu().numpy(), average='macro',
                                 zero_division=0)
        batch_war = recall_score(labels.cpu().numpy(), predictions_batch.cpu().numpy(), average='weighted',
                                 zero_division=0)
        batch_f1 = f1_score(labels.cpu().numpy(), predictions_batch.cpu().numpy(), average='weighted', zero_division=0)

        losses_meter.update(total_loss_main.item(), labels.size(0))
        acc_meter.update(batch_acc, labels.size(0))
        uar_meter.update(batch_uar, labels.size(0))
        war_meter.update(batch_war, labels.size(0))
        f1_meter.update(batch_f1, labels.size(0))

        if i % args.print_freq == 0:
            progress.display(i)
            # Optional: print individual loss components
            # print(f'  Loss_CLS: {loss_cls.item():.4f}, Loss_CON: {consistency_loss.item():.4f}, Loss_DIF: {difference_loss.item():.4f}')
            # print(f'  Loss_ADV_S: {loss_adv_shared.item():.4f}, Loss_ADV_P: {loss_adv_private.item():.4f}')
            # print(f'  Loss_D_S_Train: {loss_D_S_train.item():.4f}, Loss_D_P_Train: {loss_D_P_train.item():.4f}')

        running_loss_epoch += total_loss_main.item() * labels.size(0)  # Weighted by batch size
        all_labels_epoch.extend(labels.cpu().numpy())
        all_predictions_epoch.extend(predictions_batch.cpu().numpy())

    # Calculate average metrics for the epoch
    epoch_loss_avg = running_loss_epoch / len(all_labels_epoch) if len(all_labels_epoch) > 0 else 0
    epoch_acc_avg = accuracy_score(all_labels_epoch, all_predictions_epoch)
    epoch_uar_avg = recall_score(all_labels_epoch, all_predictions_epoch, average='macro', zero_division=0)
    epoch_war_avg = recall_score(all_labels_epoch, all_predictions_epoch, average='weighted', zero_division=0)
    epoch_f1_avg = f1_score(all_labels_epoch, all_predictions_epoch, average='weighted', zero_division=0)

    cm_epoch = confusion_matrix(all_labels_epoch, all_predictions_epoch)

    train_summary = (
        f"Epoch [{epoch}] Train Summary:\n"
        f"  Avg Loss: {epoch_loss_avg:.4f}\n"
        f"  ACC: {epoch_acc_avg:.4f}, UAR: {epoch_uar_avg:.4f}, WAR: {epoch_war_avg:.4f}, F1: {epoch_f1_avg:.4f}\n"
        f"  Classification Report:\n{classification_report(all_labels_epoch, all_predictions_epoch, digits=4, zero_division=0)}\n"
        f"  Confusion Matrix:\n{cm_epoch}\n"
        f"{'-' * 30}\n"
    )
    print(train_summary)
    with open(log_txt_path, 'a') as f:
        f.write(train_summary)

    return epoch_loss_avg, epoch_acc_avg, epoch_uar_avg, epoch_war_avg, epoch_f1_avg


# ==============================================================================
# Validation Function
# ==============================================================================
def validate(dataloader, model, criterion_cls,  # Discriminators are not trained or used for main val loss
             epoch, args, log_txt_path, device):
    losses_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('ACC', ':.4f')
    uar_meter = AverageMeter('UAR', ':.4f')
    war_meter = AverageMeter('WAR', ':.4f')
    f1_meter = AverageMeter('F1', ':.4f')
    progress = ProgressMeter(len(dataloader),
                             [losses_meter, acc_meter, uar_meter, war_meter, f1_meter],
                             prefix=f'Epoch: [{epoch}] Val  ',
                             log_txt_path=log_txt_path)

    model.eval()  # Switch to evaluation mode

    running_loss_epoch = 0.0
    all_labels_epoch = []
    all_predictions_epoch = []

    with torch.no_grad():  # Disable gradient calculations
        for i, (video, audio, labels) in enumerate(dataloader):
            video = video.to(device)
            audio = audio.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            # Forward pass
            output_logits, _, _, _, _, consistency_loss, difference_loss = model(video,
                                                                                 audio)  # Adversarial outputs not needed for val loss

            # Calculate classification loss and other relevant losses for evaluation
            loss_cls = criterion_cls(output_logits, labels)
            # Total validation loss (excluding adversarial components used for training discriminators)
            total_loss_val = loss_cls + args.lambda_sim * consistency_loss + args.lambda_diff * difference_loss

            predictions_batch = (torch.sigmoid(output_logits) > 0.5).float()

            batch_acc = accuracy_score(labels.cpu().numpy(), predictions_batch.cpu().numpy())
            batch_uar = recall_score(labels.cpu().numpy(), predictions_batch.cpu().numpy(), average='macro',
                                     zero_division=0)
            batch_war = recall_score(labels.cpu().numpy(), predictions_batch.cpu().numpy(), average='weighted',
                                     zero_division=0)
            batch_f1 = f1_score(labels.cpu().numpy(), predictions_batch.cpu().numpy(), average='weighted',
                                zero_division=0)

            losses_meter.update(total_loss_val.item(), labels.size(0))
            acc_meter.update(batch_acc, labels.size(0))
            uar_meter.update(batch_uar, labels.size(0))
            war_meter.update(batch_war, labels.size(0))
            f1_meter.update(batch_f1, labels.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

            running_loss_epoch += total_loss_val.item() * labels.size(0)
            all_labels_epoch.extend(labels.cpu().numpy())
            all_predictions_epoch.extend(predictions_batch.cpu().numpy())

    epoch_loss_avg = running_loss_epoch / len(all_labels_epoch) if len(all_labels_epoch) > 0 else 0
    epoch_acc_avg = accuracy_score(all_labels_epoch, all_predictions_epoch)
    epoch_uar_avg = recall_score(all_labels_epoch, all_predictions_epoch, average='macro', zero_division=0)
    epoch_war_avg = recall_score(all_labels_epoch, all_predictions_epoch, average='weighted', zero_division=0)
    epoch_f1_avg = f1_score(all_labels_epoch, all_predictions_epoch, average='weighted', zero_division=0)

    cm_epoch = confusion_matrix(all_labels_epoch, all_predictions_epoch)

    val_summary = (
        f"Epoch [{epoch}] Val Summary:\n"
        f"  Avg Loss: {epoch_loss_avg:.4f}\n"
        f"  ACC: {epoch_acc_avg:.4f}, UAR: {epoch_uar_avg:.4f}, WAR: {epoch_war_avg:.4f}, F1: {epoch_f1_avg:.4f}\n"
        f"  Classification Report:\n{classification_report(all_labels_epoch, all_predictions_epoch, digits=4, zero_division=0)}\n"
        f"  Confusion Matrix:\n{cm_epoch}\n"
        f"{'-' * 30}\n"
    )
    print(val_summary)
    with open(log_txt_path, 'a') as f:
        f.write(val_summary)

    return epoch_loss_avg, epoch_acc_avg, epoch_uar_avg, epoch_war_avg, epoch_f1_avg


# ==============================================================================
# Compute Final Scores and Plots
# ==============================================================================
def compute_score(dataloader, model, checkpoint_dir_path,  # Renamed checkpoint_path to checkpoint_dir_path
                  log_confusion_matrix_path, log_roc_curve_path,
                  log_txt_path, data_set_identifier, class_names, device):
    # Load the best checkpoint
    checkpoint_best_path = os.path.join(checkpoint_dir_path, 'best_model.pth')
    if os.path.exists(checkpoint_best_path):
        pre_trained_dict = torch.load(checkpoint_best_path, map_location=device)['state_dict']
        model.load_state_dict(pre_trained_dict)
        print(f"Loaded best model checkpoint from: {checkpoint_best_path}")
    else:
        print(f"Warning: Best model checkpoint not found at {checkpoint_best_path}. Using current model state.")

    model.eval()

    all_labels_list = []
    all_predictions_list = []
    all_probabilities_list = []  # To store probabilities for ROC AUC

    with torch.no_grad():
        for i, (video, audio, labels) in enumerate(tqdm.tqdm(dataloader, desc="Computing final scores")):
            video = video.to(device)
            audio = audio.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # Assuming binary classification

            output_logits, _, _, _, _, _, _ = model(video, audio)  # Only need logits for final scoring

            probabilities_batch = torch.sigmoid(output_logits)
            predictions_batch = (probabilities_batch > 0.5).float()

            all_labels_list.extend(labels.cpu().numpy())
            all_predictions_list.extend(predictions_batch.cpu().numpy())
            all_probabilities_list.extend(probabilities_batch.cpu().numpy())

    all_labels_np = np.array(all_labels_list)
    all_predictions_np = np.array(all_predictions_list)
    all_probabilities_np = np.array(all_probabilities_list)

    # Calculate metrics
    acc_score = accuracy_score(all_labels_np, all_predictions_np)
    uar_score = recall_score(all_labels_np, all_predictions_np, average='macro', zero_division=0)
    war_score = recall_score(all_labels_np, all_predictions_np, average='weighted', zero_division=0)
    f1_val_score = f1_score(all_labels_np, all_predictions_np, average='weighted', zero_division=0)
    roc_auc_val = roc_auc_score(all_labels_np, all_probabilities_np)  # Use probabilities for ROC AUC

    # Confusion matrix
    cm_final = confusion_matrix(all_labels_np, all_predictions_np)
    normalized_cm = cm_final.astype('float') / cm_final.sum(axis=1)[:, np.newaxis]
    normalized_cm = np.nan_to_num(normalized_cm) * 100  # Handle division by zero if a row sum is 0

    # Convert scores to percentage for display/logging
    acc_percent = acc_score * 100
    uar_percent = uar_score * 100
    war_percent = war_score * 100
    f1_percent = f1_val_score * 100
    roc_auc_percent = roc_auc_val * 100

    final_score_summary = (
        f"********* Final Scores for Fold {data_set_identifier} *********\n"
        f"{classification_report(all_labels_np, all_predictions_np, digits=4, zero_division=0)}\n"
        f"ACC: {acc_percent:.2f}%\n"
        f"UAR: {uar_percent:.2f}%\n"
        f"WAR: {war_percent:.2f}%\n"
        f"F1 Score: {f1_percent:.2f}%\n"
        f"ROC AUC: {roc_auc_percent:.2f}%\n"
        f"Confusion Matrix (raw):\n{cm_final}\n"
        f"Normalized Confusion Matrix (%):\n{normalized_cm}\n"
        f"{'*' * 30}\n"
    )
    print(final_score_summary)
    with open(log_txt_path, 'a') as f:
        f.write(final_score_summary)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_title = f"Confusion Matrix on Fold {data_set_identifier} (Normalized %)"
    plot_confusion_matrix(normalized_cm, classes=class_names, savepath=log_confusion_matrix_path, normalize=True,
                          title=cm_title)

    # Plot ROC curve and get FPR, TPR, AUC
    fpr_roc, tpr_roc, auc_roc = plot_roc_curve(all_labels_np, all_probabilities_np, args, log_roc_curve_path)
    with open(log_txt_path, 'a') as f:
        f.write('ROC Curve Data:\n')
        f.write(f"  AUC: {auc_roc:.4f}\n")
        # f.write(f"  FPR: {fpr_roc.tolist()}\n") # Can be very long
        # f.write(f"  TPR: {tpr_roc.tolist()}\n") # Can be very long
        f.write(f"{'*' * 30}\n")

    return acc_percent, uar_percent, war_percent, f1_percent, roc_auc_percent, fpr_roc, tpr_roc


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args)  # Set seed for reproducibility

    args.num_classes = 1  # For binary classification with BCEWithLogitsLoss
    args.class_names = ['HC', 'CI']  # Example class names for plotting, adjust as needed

    num_folds = 5  # Total number of folds for cross-validation

    # Initialize accumulators for metrics across folds
    metrics_sum = {'ACC': 0.0, 'UAR': 0.0, 'WAR': 0.0, 'F1': 0.0, 'ROC_AUC': 0.0}
    metrics_sq_sum = {'ACC': 0.0, 'UAR': 0.0, 'WAR': 0.0, 'F1': 0.0, 'ROC_AUC': 0.0}

    current_timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
    augmentation_status_str = 'UseAug' if args.use_augmentation else 'NoAug'

    # Global log file for the entire experiment (across folds)
    log_file = (  # This global variable is used in main()
        f'./Ab_loss_weight/{args.log}-{args.model}-{args.dataset}-{args.criterion}-'
        f'{args.sim_loss}-{args.diff_loss}-lambdas_{args.lambda_sim}_{args.lambda_diff}_{args.lambda_adv}-'
        f'{augmentation_status_str}-{current_timestamp}/'
    )
    os.makedirs(log_file, exist_ok=True)

    # Results summary file for all folds
    overall_results_file_path = os.path.join(log_file, "cross_validation_results.txt")
    with open(overall_results_file_path, "w") as f:  # Create/overwrite the file
        f.write(f"Cross-validation Results for Experiment: {current_timestamp}\n")
        f.write("Command line arguments:\n")
        for k, v in vars(args).items():
            f.write(f"  {k} = {v}\n")
        f.write("\n" + "*" * 40 + "\n\n")

    print('************************ Experiment Configuration ************************')
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print('************************************************************************\n')

    all_folds_fpr = []
    all_folds_tpr = []

    for fold_idx in range(num_folds):  # 0 to num_folds-1
        print(f"\n========== Starting Fold {fold_idx + 1}/{num_folds} ==========")
        # `main` function performs training and evaluation for one fold
        acc, uar, war, f1, roc_auc, fpr, tpr = main(fold_idx, args)

        metrics_sum['ACC'] += float(acc)
        metrics_sum['UAR'] += float(uar)
        metrics_sum['WAR'] += float(war)
        metrics_sum['F1'] += float(f1)
        metrics_sum['ROC_AUC'] += float(roc_auc)

        metrics_sq_sum['ACC'] += float(acc) ** 2
        metrics_sq_sum['UAR'] += float(uar) ** 2
        metrics_sq_sum['WAR'] += float(war) ** 2
        metrics_sq_sum['F1'] += float(f1) ** 2
        metrics_sq_sum['ROC_AUC'] += float(roc_auc) ** 2

        all_folds_fpr.append(fpr)
        all_folds_tpr.append(tpr)

        # Append current fold's results to the overall summary file
        with open(overall_results_file_path, "a") as f:
            f.write(
                f"Fold {fold_idx + 1:<2}: ACC = {acc:.2f}% | UAR = {uar:.2f}% | WAR = {war:.2f}% | "
                f"F1 = {f1:.2f}% | ROC AUC = {roc_auc:.2f}%\n"
            )
        print(f"========== Finished Fold {fold_idx + 1}/{num_folds} ==========\n")

    # Plot average ROC curve across folds
    multi_fold_roc_curve_path = os.path.join(log_file, "multi_fold_avg_roc_curve.png")
    plot_fold_ROC_CURVE(all_folds_fpr, all_folds_tpr,
                        multi_fold_roc_curve_path)  # Assumes this function handles averaging or plotting multiple

    # Calculate average and standard deviation for each metric
    avg_metrics = {key: val / num_folds for key, val in metrics_sum.items()}
    std_metrics = {
        key: math.sqrt(max(0, metrics_sq_sum[key] / num_folds - avg_metrics[key] ** 2))
        for key in avg_metrics
    }

    final_summary_header = '********* Cross-Validation Final Summary *********'
    print(final_summary_header)
    with open(overall_results_file_path, "a") as f:
        f.write("\n" + final_summary_header + "\n")

    for metric_name in avg_metrics:
        avg_val = avg_metrics[metric_name]
        std_val = std_metrics[metric_name]
        result_line = f"Average {metric_name + ':':<10} {avg_val:.2f}% ± {std_val:.2f}%"
        print(result_line)
        with open(overall_results_file_path, "a") as f:
            f.write(result_line + "\n")

    print('**************************************************')
    with open(overall_results_file_path, "a") as f:
        f.write("**************************************************\n")

    print(f"Detailed logs and results saved in: {log_file}")