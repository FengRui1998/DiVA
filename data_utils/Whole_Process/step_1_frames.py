import glob
import os
import cv2
import shutil
import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm import tqdm  # 进度条库
import librosa  # 音频处理库
import soundfile as sf  # 用于保存音频

'''

将原始数据进行读取视频帧
每隔5帧保存

将音频保存为wav

'''



def process_mp4_file(video_path, csv_file, output_folder, frame_interval=5, target_sample_rate=22050):
    """
    处理每个MP4文件，提取指定间隔的视频帧和音频，并将其保存到output_folder中
    target_sample_rate: 目标采样率，默认为22050
    """
    try:
        os.makedirs(output_folder, exist_ok=True)

        # 直接复制原始CSV文件
        csv_output_path = os.path.join(output_folder, 'label.csv')
        shutil.copy(csv_file, csv_output_path)

        # 提取视频帧
        video_capture = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_frame_count = 0
        frame_save_dir = os.path.join(output_folder, 'frames')
        os.makedirs(frame_save_dir, exist_ok=True)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

        with tqdm(total=total_frames, desc=f"Processing frames from {os.path.basename(video_path)}", unit="frame",
                  dynamic_ncols=True) as pbar:

            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break
                # 每隔 frame_interval 保存一次
                if frame_count % frame_interval == 0:
                    frame_file = os.path.join(frame_save_dir, f'frame_{saved_frame_count:04d}.jpg')
                    cv2.imwrite(frame_file, frame)
                    saved_frame_count += 1
                frame_count += 1
                pbar.update(1)
        video_capture.release()

        # 提取音频并保存为WAV文件
        audio_output_path = os.path.join(output_folder, 'audio.wav')
        video_clip = VideoFileClip(video_path)
        temp_audio_path = os.path.join(output_folder, 'temp_audio.wav')
        video_clip.audio.write_audiofile(temp_audio_path)  # 先保存临时音频
        video_clip.close()

        # 使用 librosa 加载音频并调整采样率为 target_sample_rate
        audio_data, original_sr = librosa.load(temp_audio_path, sr=None)  # 保持原采样率加载
        resampled_audio = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sample_rate)  # 正确传入参数

        sf.write(audio_output_path, resampled_audio, target_sample_rate)  # 保存为目标采样率的音频文件

        # 删除临时音频文件
        os.remove(temp_audio_path)



    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")


def process_all_data(root_dir, output_base_dir):
    """
    遍历所有病例文件夹，处理每个MP4文件
    """
    categories = ['AD', 'MCI', 'NC']

    for category in categories:
        category_path = os.path.join(root_dir, category)

        for case_folder in tqdm(os.listdir(category_path), desc=f"Processing category {category}", unit="folder"):
            case_path = os.path.join(category_path, case_folder)
            if os.path.isdir(case_path):
                # 查找所有的MP4文件
                video_files = [f for f in os.listdir(case_path) if f.endswith('.mp4')]
                csv_file = glob.glob(os.path.join(case_path, '*.csv'))

                if csv_file:
                    csv_file = csv_file[0]  # 假设每个病例文件夹只有一个CSV
                    for i, video_file in enumerate(tqdm(video_files, desc=f"Processing {case_folder}", leave=False, unit="file")):
                        video_path = os.path.join(case_path, video_file)
                        # 创建新的输出文件夹，名称为“病例文件名 + _video_ + 数字”
                        reference_path = os.path.join(output_base_dir, category)
                        os.makedirs(reference_path, exist_ok=True)

                        output_folder = os.path.join(reference_path, case_folder, f"{case_folder}_video_{i + 1:03d}")
                        process_mp4_file(video_path, csv_file, output_folder, frame_interval=5, target_sample_rate=22050)
                else:
                    print(f"No CSV file found in {case_folder}")


# 定义保存路径
output_base_dir = 'G:\MCI_project\Dataset\DATA_All_processed'
os.makedirs(output_base_dir, exist_ok=True)
# 使用该函数处理数据集
root_dir = 'G:\MCI_project\Dataset\DATA_All'
process_all_data(root_dir, output_base_dir)
