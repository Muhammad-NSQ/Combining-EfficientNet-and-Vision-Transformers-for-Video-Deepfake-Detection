import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from cross_efficient_vit import CrossEfficientViT
from utils import (
    get_method, check_correct, resize, shuffle_dataset, get_n_params,
    transform_frame, custom_round, custom_video_round
)
import glob
from os import cpu_count
import json
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, 
    HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, 
    PadIfNeeded, GaussNoise, GaussianBlur, Rotate
)
from transforms.albu import IsotropicResize
import yaml
import argparse

#########################
####### CONSTANTS #######
#########################

MODELS_DIR = Path("models")
BASE_DIR = Path("E:/DF_dataset/deep_fakes") # TODO: change for linux
DATA_DIR = BASE_DIR / "dataset"
TEST_DIR = DATA_DIR / "test_set"
OUTPUT_DIR = MODELS_DIR / "tests"
TEST_LABELS_PATH = DATA_DIR / "dfdc_test_labels.csv"

#########################
####### UTILITIES #######
#########################

def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["original", "fake"])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig(OUTPUT_DIR / "confusion.jpg")
    plt.close(fig)

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    fpr, tpr, th = metrics.roc_curve(correct_labels, preds)
    model_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Model_{model_name} (area = {model_auc:.3f})")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(OUTPUT_DIR / f"{model_name}_{opt.dataset}_acc{accuracy*100:.2f}_loss{loss:.2f}_f1{f1:.2f}.jpg")
    plt.clf()

def read_frames(video_path, videos, opt, config):
    method = get_method(video_path, str(DATA_DIR))
    if "Original" in video_path:
        label = 0.
    elif method == "DFDC":
        test_df = pd.read_csv(TEST_LABELS_PATH)
        video_folder_name = os.path.basename(video_path)
        video_key = f"{video_folder_name}.mp4"
        label = test_df.loc[test_df['filename'] == video_key, 'label'].values[0]
    else:
        label = 1.

    frames_number = len(os.listdir(video_path))
    frames_interval = int(frames_number / opt.frames_per_video)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {i: [] for i in range(3)}  # Consider up to 3 faces per video

    for path in frames_paths:
        for i in range(3):
            if f"_{i}" in path:
                frames_paths_dict[i].append(path)

    if frames_interval > 0:
        for key in frames_paths_dict:
            if len(frames_paths_dict[key]) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            frames_paths_dict[key] = frames_paths_dict[key][:opt.frames_per_video]

    video = {}
    for key, frame_images in frames_paths_dict.items():
        transform = create_base_transform(config['model']['image-size'])
        for frame_image in frame_images:
            image = transform(image=cv2.imread(os.path.join(video_path, frame_image)))['image']
            if len(image) > 0:
                if key in video:
                    video[key].append(image)
                else:
                    video[key] = [image]
    videos.append((video, label, video_path))


def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

#########################
#######   MAIN    #######
#########################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=10, type=int, help='Number of data loader workers.')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH', help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='DFDC', help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)")
    parser.add_argument('--max_videos', type=int, default=-1, help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=30, help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    
    opt = parser.parse_args()
    print(opt)
    
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
        
    if os.path.exists(opt.model_path):
        model = CrossEfficientViT(config=config)
        model.load_state_dict(torch.load(opt.model_path))
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    model_name = Path(opt.model_path).name

    OUTPUT_DIR = OUTPUT_DIR / opt.dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mgr = Manager()
    paths = []
    videos = mgr.list()

    folders = ["Original", opt.dataset] if opt.dataset != "DFDC" else [opt.dataset]

    # Read all videos paths
    for folder in folders:
        method_folder = TEST_DIR / folder
        for video_folder in os.listdir(method_folder):
            paths.append(method_folder / video_folder)

    # Read faces
    with Pool(processes=cpu_count()-1) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, opt=opt, config=config, videos=videos), [str(path) for path in paths]):
                pbar.update()


    video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[1] for row in videos])
    videos = np.asarray([row[0] for row in videos])
    preds = []

    # Perform prediction
    bar = Bar('Predicting', max=len(videos))

    with open(OUTPUT_DIR / f"{opt.dataset}_{model_name}_labels.txt", "w+") as f:
        for index, video in enumerate(videos):
            video_faces_preds = []
            video_name = video_names[index]
            f.write(str(video_name))
            for key in video:
                faces_preds = []
                video_faces = video[key]
                for i in range(0, len(video_faces), opt.batch_size):
                    faces = video_faces[i:i+opt.batch_size]
                    faces = torch.tensor(np.asarray(faces))
                    if faces.shape[0] == 0:
                        continue
                    faces = np.transpose(faces, (0, 3, 1, 2))
                    faces = faces.cuda().float()
                    
                    pred = model(faces)
                    
                    scaled_pred = [torch.sigmoid(p) for p in pred]
                    faces_preds.extend(scaled_pred)
                
                current_faces_pred = sum(faces_preds)/len(faces_preds)
                face_pred = current_faces_pred.cpu().detach().numpy()[0]
                f.write(f" {face_pred}")
                video_faces_preds.append(face_pred)
            bar.next()
            video_pred = custom_video_round(video_faces_preds) if len(video_faces_preds) > 1 else video_faces_preds[0]
            preds.append([video_pred])
            
            f.write(f" --> {video_pred} (CORRECT: {correct_test_labels[index]})\n")
    
    bar.finish()

    #########################
    #######  METRICS  #######
    #########################

    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor(preds)

    loss = loss_fn(tensor_preds, tensor_labels).item()
    accuracy = accuracy_score(custom_round(np.asarray(preds)), correct_test_labels)
    f1 = f1_score(correct_test_labels, custom_round(np.asarray(preds)))
    print(f"{model_name} Test Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, F1: {f1:.4f}")
    save_roc_curves(correct_test_labels, preds, model_name, accuracy, loss, f1)
    save_confusion_matrix(correct_test_labels, custom_round(np.asarray(preds)))