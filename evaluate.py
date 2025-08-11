import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
import argparse
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import sys
sys.path.append(os.path.abspath('data_loader'))
from dataset_loader import get_data_loaders
from model.auxformer.avmodel_AF import AVmodel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import datetime
from sklearn.metrics import roc_curve,roc_auc_score, confusion_matrix
import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Combine')
    parser.add_argument('--model', type=str, default='AuxFormer')
    parser.add_argument('--data_loader', type=str, default='Clip', choices=['Clip', 'Full'],
                        help='dataset Clip or Full')
    parser.add_argument('--use_augmentation', type=bool, default=True, help='use data augmentation during training')
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--print-freq', type=int, default=5)
    parser.add_argument('--criterion', type=str, default='BCE', choices=['BCE', 'Focal'])
    parser.add_argument('--checkpoint', type=str,
                        default='/public/home/feng_rui/code/MCI_project/log/Clip-AV-Auxformer-Combine-Cls-2-Epoch20-Batchsize8-UseAugmentation-2412161721-log/Combine-set5-log/best_model.pth')
    parser.add_argument('--set', type=int, default=5)
    parser.add_argument('--img-size', type=int, default=224)

    args = parser.parse_args()
    return args

def main(set, args):
    
    data_set = set
    
    if args.dataset == "MCI":
        print("*********** MCI Dataset Fold  " + str(data_set) + " ***********")

    log_txt_path = log_file + 'log.txt'
    log_confusion_matrix_path = log_file + 'cn.png'
    log_roc_curve_path = log_file + 'ROC.png'

    train_loader, val_loader = get_data_loaders(args, data_set)

    model = AVmodel(args=args).cuda()

 
    acc, uar, war, f1, roc_auc, fpr, tpr = computer_score(val_loader, model, args.checkpoint, log_confusion_matrix_path, log_roc_curve_path, log_txt_path, data_set, args.class_names)
  
    return acc, uar, war, f1, roc_auc, fpr, tpr

def computer_score(dataloader, model, checkpoint_path, log_confusion_matrix_path,log_roc_curve_path, log_txt_path, data_set, class_names):
    #加载checkpoint
    checkpoint_best = checkpoint_path
    pre_trained_dict = torch.load(checkpoint_best)['state_dict']
    model.load_state_dict(pre_trained_dict)

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []  
    with torch.no_grad():
        for i, (video, audio, labels) in enumerate(tqdm.tqdm(dataloader)):
            video = video.cuda()
            audio = audio.cuda()
            labels = labels.float().unsqueeze(1).cuda()
            # compute output
            output = model(video, audio)


            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()


            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())


    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)


    acc = accuracy_score(all_labels, all_predictions)
    uar = recall_score(all_labels, all_predictions, average='macro')
    war = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')


    roc_auc = roc_auc_score(all_labels, all_probabilities)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    np.set_printoptions(precision=4)
    normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100

    acc = 100 * acc
    uar = 100 * uar
    war = 100 * war
    f1 = 100 * f1
    roc_auc = 100 * roc_auc


    print(f"ACC: {acc:.2f}")
    print(f"UAR: {uar:.2f}")
    print(f"WAR: {war:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print("Confusion Matrix:\n", cm)

    with open(log_txt_path, 'a') as f:
        f.write('************************\n')
        f.write(f"ACC: {acc:.2f}\n")
        f.write(f"UAR: {uar:.2f}\n")
        f.write(f"WAR: {war:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")
        f.write(f"ROC AUC: {roc_auc:.2f}\n")
        f.write("Confusion Matrix:\n")


        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")
        f.write('************************\n')

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))

    title_ = "Confusion Matrix on fold " + str(data_set)

    plot_confusion_matrix(normalized_cm, classes=class_names, savepath=log_confusion_matrix_path, normalize=True, title=title_)

    # 调用 plot_roc_curve 函数
    fpr, tpr, auc = plot_roc_curve(all_labels, all_probabilities, args, log_roc_curve_path)
    # 打开日志文件，写入 ROC 数据
    # with open(log_txt_path, 'a') as f:
    #     f.write('************************\n')
    #     f.write('ROC Curve Data\n')
    #     f.write('************************\n')


    #     f.write(f"AUC: {auc:.3f}\n")
    #     f.write(f"FPR: {fpr:.5f}\n")
    #     f.write(f"TPR: {tpr:.5f}\n")
    #     f.write('************************\n')


    with open(log_txt_path, 'a') as f:
        f.write('************************\n')
        f.write('ROC Curve Data2\n')
        f.write('************************\n')


        f.write(f"AUC: {auc:.3f}\n")


        f.write(f"FPR: {' '.join([f'{x:.5f}' for x in fpr])}\n")
        f.write(f"TPR: {' '.join([f'{x:.5f}' for x in tpr])}\n")
        f.write('************************\n')



    return acc, uar, war, f1, roc_auc, fpr, tpr

if __name__ == '__main__':
    args = parse_args()
    now = datetime.datetime.now()
    time_str = now.strftime("%y%m%d%H%M")
    print('************************')
    args.num_classes = 1
    args.class_names = ['HC.', 'CI.']

    log_file = './Evaluate/' + 'Evaluate-' + str(args.dataset)+ '-set-' + str(args.set)+ '-' + time_str+ '-log/'
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    acc, uar, war, f1, roc_auc, fpr, tpr = main(args.set, args)
