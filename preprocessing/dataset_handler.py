import argparse
import json
import os
import numpy as np
from typing import Type
import cv2
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import face_detector
from face_detector import VideoDataset, VideoFaceDetector
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import random
from PIL import Image
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pil_to_numpy(pil_img):
    """Convert a PIL Image to a NumPy array."""
    return np.array(pil_img)

def enhance_image(image):
    """Apply enhancements to improve image quality."""
    # Sharpening filter kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def process_video(video_path, detector, output_path, crop_size, sets_distribution, face_type='face'):
    try:
        detector = detector(device="cuda:0")
        dataset = VideoDataset([video_path])
        loader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1, collate_fn=lambda x: x)
        
        for item in loader:
            video, indices, frames = item[0]
            id = Path(video).stem
            
            batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
            
            all_crops = []
            for j, batch_frames in enumerate(batches):
                try:
                    # Ensure all frames are numpy arrays in BGR format, suitable for OpenCV processing
                    numpy_batch_frames = []
                    for frame in batch_frames:
                        if isinstance(frame, Image.Image):
                            frame = pil_to_numpy(frame)
                        # Ensure frame is in BGR format for OpenCV processing
                        if frame.shape[2] == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        numpy_batch_frames.append(frame)

                    # Perform face detection (assuming the detector expects BGR input)
                    batch_results = detector._detect_faces(numpy_batch_frames)
                except Exception as e:
                    logging.error(f"Error detecting faces in batch {j} of video {video_path}: {str(e)}")
                    continue

                for i, (frame, bboxes) in enumerate(zip(numpy_batch_frames, batch_results)):
                    frame_idx = j * detector._batch_size + i
                    if bboxes is not None:
                        for bbox_idx, bbox in enumerate(bboxes):
                            try:
                                if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) == 4:
                                    xmin, ymin, xmax, ymax = [int(b) for b in bbox]
                                    
                                    # Expand bbox based on face_type
                                    if face_type == 'head':
                                        h = ymax - ymin
                                        w = xmax - xmin
                                        xmin = max(0, xmin - int(w * 0.3))
                                        ymin = max(0, ymin - int(h * 0.3))
                                        xmax = min(frame.shape[1], xmax + int(w * 0.3))
                                        ymax = min(frame.shape[0], ymax + int(h * 0.2))
                                    elif face_type == 'full_face':
                                        h = ymax - ymin
                                        w = xmax - xmin
                                        xmin = max(0, xmin - int(w * 0.15))
                                        ymin = max(0, ymin - int(h * 0.15))
                                        xmax = min(frame.shape[1], xmax + int(w * 0.15))
                                        ymax = min(frame.shape[0], ymax + int(h * 0.15))
                                    
                                    crop = frame[ymin:ymax, xmin:xmax]
                                    if crop.size != 0:
                                        # Enhance and resize the crop
                                        crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
                                        crop_resized = enhance_image(crop_resized)
                                        all_crops.append((frame_idx, bbox_idx, cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)))
                                else:
                                    logging.warning(f"Invalid bbox format in frame {frame_idx}, skipping")
                            except Exception as e:
                                logging.error(f"Error processing bbox in frame {frame_idx} of video {video_path}: {str(e)}")
                                continue
            
            # Determine which set this video belongs to
            rand_val = random.random()
            if rand_val < sets_distribution[0]:
                set_name = "training_set"
            elif rand_val < sets_distribution[0] + sets_distribution[1]:
                set_name = "test_set"
            else:
                set_name = "validation_set"
            
            output_dir = Path(output_path) / "dataset" / set_name / Path(video_path).parent.name / id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for frame_idx, bbox_idx, crop in all_crops:
                # Save crop as RGB
                cv2.imwrite(str(output_dir / f"{frame_idx}_{bbox_idx}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str, help='Dataset directory')
    parser.add_argument('--output_path', default=os.getcwd(), type=str, help='Output directory (default: current directory)')
    parser.add_argument('--crop_size', default=228, type=int, help='Size of face crops')
    parser.add_argument('--detector_type', default="FacenetDetector", choices=["FacenetDetector"], help='Type of face detector')
    parser.add_argument('--train_ratio', default=0.8, type=float, help='Ratio of training set')
    parser.add_argument('--test_ratio', default=0.15, type=float, help='Ratio of test set')
    parser.add_argument('--face_type', default='face', choices=['face', 'full_face', 'head'], help='Type of face extraction')
    opt = parser.parse_args()

    validation_ratio = 1 - opt.train_ratio - opt.test_ratio
    sets_distribution = (opt.train_ratio, opt.test_ratio, validation_ratio)

    detector_cls = face_detector.__dict__[opt.detector_type]

    video_paths = []
    for root, dirs, files in os.walk(opt.data_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_paths.append(os.path.join(root, file))

    with Pool(processes=cpu_count()-2) as p:
        with tqdm(total=len(video_paths)) as pbar:
            for _ in p.imap_unordered(
                partial(process_video, 
                        detector=detector_cls, 
                        output_path=opt.output_path, 
                        crop_size=opt.crop_size, 
                        sets_distribution=sets_distribution,
                        face_type=opt.face_type),
                video_paths
            ):
                pbar.update()

if __name__ == "__main__":
    main()
