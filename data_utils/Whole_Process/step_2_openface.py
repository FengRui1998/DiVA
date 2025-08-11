import glob
import os
import shutil
import subprocess
import glob
import os
import shutil
import subprocess
import time
import cv2
import soundfile as sf
from facenet_pytorch import MTCNN
import cv2
from moviepy.editor import VideoFileClip
import numpy as np
from tqdm import tqdm
import os
import numpy as np
import moviepy.editor as mp
import librosa
import noisereduce as nr
from scipy.signal import butter, lfilter

'''

对保存的视频帧进行openface 扣取人脸

'''



# OpenFace 可执行文件路径
PATH_TO_OPENFACE_Win = "D:\\FR_MCI_Project\\OpenFace_2.2.0_win_x64"

detector = MTCNN()
def is_valid_face(image_path, confidence_threshold=0.9):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用 PyTorch 版本的 MTCNN 进行检测
    boxes, probs = detector.detect(image_rgb)

    if probs is not None:
        for prob in probs:
            if prob is not None and prob > confidence_threshold:
                return True
    return False

def generate_face_images(input_root, save_root):

    landmark_path = glob.glob(os.path.join(input_root, '*.csv'))
    # 获取 input_root 的上级目录
    parent_directory = os.path.abspath(os.path.join(input_root, os.pardir))
    # 获取 CSV 文件的名称
    csv_file_name = os.path.basename(landmark_path[0])
    # 构建目标文件路径
    destination_path = os.path.join(parent_directory, csv_file_name)
    shutil.copy(landmark_path[0], destination_path)


    all_frame_names = []  # 存储所有的frame_names
    dir_paths = sorted(glob.glob(os.path.join(input_root, '*_aligned')))
    # 获取所有的frame_names
    for dir_path in dir_paths:
        frame_names = sorted(os.listdir(dir_path))
        all_frame_names.extend([(dir_path, frame_name) for frame_name in frame_names])

    # 根据sample_indices抽取样本并保存
    for idx, (dir_path, frame_name) in enumerate(all_frame_names):
        frame_path = os.path.join(dir_path, frame_name)
        new_frame_name = 'frame'+ f'_{idx + 1:04d}.jpg'  # 根据需要重新命名
        save_path = os.path.join(save_root, new_frame_name)
        shutil.copy(frame_path, save_path)

        # 二次检查
        if not is_valid_face(save_path):  # 检查新保存的图片
            os.remove(save_path)  # 删除无效的人脸图片
            print(f"Removed invalid face image: {save_path}")

def process_face_images(frames_folder, output_folder, temp_folder1):
    """
    从 frames_folder 读取图像，使用 OpenFace 提取面部并保存到 output_folder。
    """

    exe_path = os.path.join(PATH_TO_OPENFACE_Win, 'FeatureExtraction.exe')

    command = [
        exe_path,
        '-fdir', frames_folder,
        '-out_dir', temp_folder1,
        '-simalign',
        '-simsize', '224',
        '-3Dfp',  # 生成 3D 特征点坐标
        '-aus',  # 提取面部动作单元 (AUs)
    ]

    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Processing of {frames_folder} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {frames_folder}: {e}")

    generate_face_images(temp_folder1, output_folder)


def process_all_face_images(root_dir):
    """
    遍历所有生成的文件夹，处理每个 frames 文件夹。
    """
    categories = ['AD', 'MCI', 'NC']

    for category in categories:
        category_path = os.path.join(root_dir, category)

        for case_folder in os.listdir(category_path):
            case_path = os.path.join(category_path, case_folder)
            # 进入 case_folder 里面的子文件夹
            for subfolder in os.listdir(case_path):
                subfolder_path = os.path.join(case_path, subfolder)

                frames_folder = os.path.join(subfolder_path, 'frames')
                output_folder = os.path.join(subfolder_path, 'frames_face')
                # 定义临时文件夹
                temp_folder1 = os.path.join(subfolder_path, 'temp1')

                if os.path.isdir(frames_folder):
                    os.makedirs(output_folder, exist_ok=True)
                    process_face_images(frames_folder, output_folder, temp_folder1)
                    shutil.rmtree(temp_folder1)

if __name__ == '__main__':
    # 定义基础文件夹
    root_dir = r'G:\MCI_project\Dataset\DATA_All_processed'
    process_all_face_images(root_dir)
