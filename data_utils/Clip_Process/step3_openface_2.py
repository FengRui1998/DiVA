import glob
import os
import shutil
import subprocess
import time
import cv2
import soundfile as sf
from facenet_pytorch import MTCNN
import cv2
from tqdm import tqdm
import os
import numpy as np
import moviepy.editor as mp
import librosa
import noisereduce as nr
from scipy.signal import butter, lfilter

# PATH_TO_FFMPEG_Win = '"C:\\Program Files\\ffmpeg-4.3.1-2021-01-01-essentials_build\\bin\\ffmpeg.exe"'
PATH_TO_OPENFACE_Win = "D:\\FR_MCI_Project\\OpenFace_2.2.0_win_x64"
FRAME_COUNT = 32
set_min_total_frames = 32
CONFIDENCE_THRESHOLD = 0.95  # 置信度阈值

detector = MTCNN()

# 你定义的音频读取和调整函数
def read_and_adjust_audio(data, target_length_seconds=6, target_sr=22050):
    # 获取当前音频的长度（以样本为单位）
    current_length_samples = data.shape[0]

    # 计算目标长度的样本数
    target_length_samples = target_length_seconds * target_sr

    # 调整音频长度
    if current_length_samples < target_length_samples:
        # 填充
        padding_length = target_length_samples - current_length_samples
        padding = np.zeros(padding_length)
        data = np.concatenate((data, padding))
    elif current_length_samples > target_length_samples:
        # 截断
        data = data[:target_length_samples]

    return data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def _get_sample_indices(num_frames, num_samples):
    if num_frames > num_samples:
        tick = num_frames / float(num_samples)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_samples)])
    else:
        offsets = np.pad(np.array(list(range(num_frames))), (0, num_samples - num_frames), 'edge')
    return offsets


def is_valid_face(image_path, confidence_threshold=CONFIDENCE_THRESHOLD):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用 PyTorch 版本的 MTCNN 进行检测
    boxes, probs = detector.detect(image_rgb)

    if probs is not None:
        for prob in probs:
            if prob is not None and prob > confidence_threshold:
                return True
    return False


def generate_face_images(input_root, save_root, num_samples=FRAME_COUNT):
    all_frame_names = []  # 存储所有的frame_names
    dir_paths = sorted(glob.glob(os.path.join(input_root, '*_aligned')))
    # 获取所有的frame_names
    for dir_path in dir_paths:
        frame_names = sorted(os.listdir(dir_path))

        # 只添加第一个图像文件
        if frame_names:
            all_frame_names.append((dir_path, frame_names[0]))


    valid_frames = [
        frame for frame in all_frame_names
        if is_valid_face(os.path.join(frame[0], frame[1]))  # 再次检查每个对齐图像的有效性
    ]

    total_frames = len(valid_frames)

    if total_frames < set_min_total_frames:
        print(f"Warning: {input_root} contains fewer than set valid frames nums. Skipping this file.")
        return False  # 跳过处理并返回 False

    sample_indices = _get_sample_indices(total_frames, num_samples)

    # 根据sample_indices抽取样本并保存
    for idx, sample_idx in enumerate(sample_indices):
        dir_path, frame_name = valid_frames[sample_idx]
        frame_path = os.path.join(dir_path, frame_name)
        new_frame_name = os.path.basename(dir_path)[:-len('_aligned')] + f'_{idx + 1}.jpg'  # 根据需要重新命名
        save_path = os.path.join(save_root, new_frame_name)
        shutil.copy(frame_path, save_path)

    return True

def extract_faces(A_folder, B_folder, temp_folder1, temp_folder2):
    for class_dir in os.listdir(A_folder):
        class_dir_path = os.path.join(A_folder, class_dir)
        for subdir, _, files in os.walk(class_dir_path):
            relative_path = os.path.relpath(subdir, A_folder)

            for file in files:
                if file.endswith('.mp4'):
                    A_file_path = os.path.join(subdir, file)
                    video_name = os.path.splitext(file)[0]
                    B_subdir = os.path.join(B_folder, relative_path, video_name)
                    temp_subdir1 = os.path.join(temp_folder1, relative_path, video_name)
                    temp_subdir2 = os.path.join(temp_folder2, relative_path, video_name)

                    audio_path = os.path.join(B_subdir, video_name + '.wav')
                    if os.path.exists(audio_path):
                        print(f"Skipping already processed directory: {B_subdir}")
                        continue

                    if not os.path.exists(B_subdir):
                        os.makedirs(B_subdir)
                    if not os.path.exists(temp_subdir1):
                        os.makedirs(temp_subdir1)
                    if not os.path.exists(temp_subdir2):
                        os.makedirs(temp_subdir2)

                    # 打开视频文件
                    cap = cv2.VideoCapture(A_file_path)
                    if not cap.isOpened():
                        print(f"错误: 无法打开视频文件 '{A_file_path}'")
                        continue

                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    pbar = tqdm(total=frame_count, desc=f"处理视频 {file}", unit="帧")

                    frame_number = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # 生成每帧的文件名
                        frame_filename = os.path.join(temp_subdir1, f"frame_{frame_number:04d}.jpg")

                        # 保存帧图像
                        if cv2.imwrite(frame_filename, frame):
                            pbar.set_postfix_str(f"成功保存帧 {frame_number} 到 {frame_filename}")
                        else:
                            print(f"错误: 无法保存帧 {frame_number} 到 {frame_filename}")

                        frame_number += 1
                        pbar.update(1)
                    pbar.close()

                    # 释放视频捕获对象
                    cap.release()
                    print(f"视频 {A_file_path} 处理完成，共提取 {frame_number} 帧图像并保存到 {temp_subdir1}")
                    #temp_subdir1 里是所有的视频帧

                    exe_path = os.path.join(PATH_TO_OPENFACE_Win, 'FaceLandmarkImg.exe')

                    command = [
                        exe_path,
                        '-fdir', temp_subdir1,
                        '-out_dir', temp_subdir2,
                        '-simalign',
                        '-simsize', '224'
                    ]

                    subprocess.run(command, check=True)
                    print(f"Processing of {temp_subdir1} completed successfully.")
                    # generate_face_images(temp_subdir2, B_subdir, num_samples=FRAME_COUNT)
                    if not generate_face_images(temp_subdir2, B_subdir, num_samples=FRAME_COUNT):
                        # 删除生成的文件夹
                        shutil.rmtree(B_subdir)
                        shutil.rmtree(temp_subdir1)
                        shutil.rmtree(temp_subdir2)
                        continue

                    # 构建 OpenFace 命令行参数
                    exe_path2 = os.path.join(PATH_TO_OPENFACE_Win, 'FeatureExtraction.exe')
                    command = [
                        exe_path2,
                        '-fdir', B_subdir,  # 如果是图像文件夹使用 -fdir
                        '-out_dir', B_subdir,
                        '-3Dfp',  # 生成 3D 特征点坐标
                        '-aus',  # 提取面部动作单元 (AUs)
                    ]

                    # 运行 OpenFace 工具
                    subprocess.run(command, check=True)
                    print(f"OpenFace 处理完成。结果已保存至 {B_subdir}")
                    # 删除对应的 TXT 文件
                    for txt_file in glob.glob(B_subdir + '/*.txt'):
                        os.remove(txt_file)

                    # 打开视频文件
                    video_ = mp.VideoFileClip(A_file_path)
                    audio = video_.audio
                    sr = 22050
                    audio_array = np.array(audio.to_soundarray(fps=sr))
                    # 将音频数据转换为单通道（如果是立体声）
                    if audio_array.shape[1] > 1:
                        audio_array = np.mean(audio_array, axis=1)

                    # 后续处理步骤
                    y = nr.reduce_noise(y=audio_array, sr=sr)
                    y = bandpass_filter(y, lowcut=300, highcut=3400, fs=sr, order=6)
                    # 在这里调用你定义的函数来调整音频长度
                    y = read_and_adjust_audio(y, target_length_seconds=6, target_sr=sr)

                    # 音量归一化
                    y = librosa.util.normalize(y)

                    # 保存处理后的音频
                    sf.write(audio_path, y, sr)

                    # cmd = '"%s -loglevel quiet -y -i %s -ar 22050 -ac 1 %s"' % (PATH_TO_FFMPEG_Win, A_file_path, audio_path)  # windows
                    # os.system(cmd)

                    # 检查并复制CSV文件
                    csv_files = [f for f in files if f.endswith('.csv')]
                    if csv_files:
                        csv_file_path = os.path.join(subdir, csv_files[0])
                        shutil.copy(csv_file_path, B_subdir)

                    shutil.rmtree(temp_subdir1)
                    shutil.rmtree(temp_subdir2)


if __name__ == '__main__':
    # 定义基础文件夹
    A_folder = r'G:\MCI_project\Dataset\tempp_1'
    B_folder = r'G:\MCI_project\Dataset\DATA_Clip_split_check_face'

    # 定义临时文件夹
    temp_folder1 = os.path.join(B_folder, 'temp1')
    temp_folder2 = os.path.join(B_folder, 'temp2')

    if not os.path.exists(temp_folder1):
        os.makedirs(temp_folder1)
    if not os.path.exists(temp_folder2):
        os.makedirs(temp_folder2)

    # 开始计时
    start_time = time.time()
    extract_faces(A_folder, B_folder, temp_folder1, temp_folder2)
    print('所有视频帧提取成功。')
    # 结束计时
    end_time = time.time()

    # 计算并打印所花费的时间
    elapsed_time = end_time - start_time
    print(f"代码执行时间: {elapsed_time} 秒")

    # 删除临时文件夹
    shutil.rmtree(temp_folder1)
    shutil.rmtree(temp_folder2)
