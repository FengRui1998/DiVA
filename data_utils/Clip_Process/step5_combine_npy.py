import os
import numpy as np
import pandas as pd
import cv2  # 用于读取图像
import glob
from scipy.io import wavfile  # 用于读取wav文件
from tqdm import tqdm

# 设置根目录
root_dir = r'G:\MCI_project\Dataset\DATA_Clip_split_check_face'
output_dir = r'G:\MCI_project\Dataset\DATA_Clip_split_check_face_npy'  # 保存npy文件的目录
os.makedirs(output_dir, exist_ok=True)

# 定义教育水平的映射字典
edu_mapping = {
    '小学': 0,
    '初中': 1,
    '高中': 2,
    '大专及以上': 3
}

# 遍历每个主要文件夹
for main_folder in tqdm(['AD', 'MCI', 'NC'], desc="主文件夹"):
    main_folder_path = os.path.join(root_dir, main_folder)

    # 遍历每个病例文件夹
    for case_folder in os.listdir(main_folder_path):
        case_folder_path = os.path.join(main_folder_path, case_folder)

        # 遍历每个子文件夹
        for sub_folder in os.listdir(case_folder_path):
            sub_folder_path = os.path.join(case_folder_path, sub_folder)

            # 检查是否是文件夹
            if os.path.isdir(sub_folder_path):
                # 读取JPG图像
                jpg_files = sorted(glob.glob(os.path.join(sub_folder_path, 'frame_*.jpg')))
                images = [cv2.imread(jpg_file) for jpg_file in jpg_files]

                # 读取WAV文件中的音频数据
                wav_file = os.path.join(sub_folder_path, 'audio.wav')
                if os.path.exists(wav_file):
                    sample_rate, audio_data = wavfile.read(wav_file)  # 读取音频采样率和数据
                else:
                    sample_rate, audio_data = None, None
                    print(f"Warning: audio file not found in {sub_folder_path}")

                # 读取Data.csv
                data_csv = os.path.join(sub_folder_path, 'Person_Data.csv')
                try:
                    data_df = pd.read_csv(data_csv, encoding='gbk')
                except UnicodeDecodeError:
                    try:
                        data_df = pd.read_csv(data_csv, encoding='utf-8')
                    except UnicodeDecodeError:
                        data_df = pd.read_csv(data_csv, encoding='latin1')

                # 提取数据并转换格式
                age = np.array(data_df['年龄'].values[0], dtype=np.int32)
                education = np.array(edu_mapping.get(data_df['学历'].values[0], -1), dtype=np.int32)

                # 处理MMSE值
                if 'MMSE' in data_df.columns and not data_df['MMSE'].isnull().all():
                    mmse_data = np.array(data_df['MMSE'].fillna(-1).values, dtype=np.int32)
                else:
                    mmse_data = np.array([-1], dtype=np.int32)

                # 处理MoCA值
                moca = np.array(data_df['MoCA'].values[0], dtype=np.int32)

                # 读取frames.csv文件中读取标签
                landmark_file = os.path.join(sub_folder_path, 'Openface_Label.csv')
                # 尝试读取CSV文件并处理编码问题
                try:
                    df_landmark = pd.read_csv(landmark_file, encoding='gbk')
                except UnicodeDecodeError:
                    try:
                        df_landmark = pd.read_csv(landmark_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        df_landmark = pd.read_csv(landmark_file, encoding='latin1')

                # 去除列名前后的空格
                df_landmark.columns = df_landmark.columns.str.strip()
                # 提取 X, Y, Z 轴的关键点列，并确保列名的顺序一致
                x_columns = [f"X_{i}" for i in range(68)]
                y_columns = [f"Y_{i}" for i in range(68)]
                z_columns = [f"Z_{i}" for i in range(68)]

                # 检查是否成功找到全部 X, Y, Z 坐标列
                missing_columns = set(x_columns + y_columns + z_columns) - set(df_landmark.columns)
                if missing_columns:
                    print(f"Warning: Missing columns in Openface_Label.csv: {missing_columns}")

                # 将 X, Y, Z 坐标列数据提取为 NumPy 数组，并用 0 填充缺失值
                x_landmarks = df_landmark[x_columns].fillna(0).values  # 用 0 填充缺失值
                y_landmarks = df_landmark[y_columns].fillna(0).values
                z_landmarks = df_landmark[z_columns].fillna(0).values

                # 将 X, Y, Z 坐标沿最后一个维度进行拼接，得到 (N, 204) 形状
                facial_landmarks = np.concatenate([x_landmarks, y_landmarks, z_landmarks], axis=1)

                # 创建子目录以保持层次结构
                relative_path = os.path.relpath(sub_folder_path, root_dir)
                output_sub_folder = os.path.join(output_dir, os.path.dirname(relative_path))
                os.makedirs(output_sub_folder, exist_ok=True)

                # 创建保存数据的字典
                data_dict = {
                    'images': images,
                    'audio_data': audio_data,  # 保存音频数据
                    'sample_rate': sample_rate,  # 保存音频采样率
                    'age': age,
                    'education': education,
                    'mmse': mmse_data,
                    'moca': moca,
                    'facial_landmarks': facial_landmarks
                }

                # 保存为npy文件
                sub_file = sub_folder.rsplit('_', 1)[0]
                npy_file_name = os.path.join(output_sub_folder, f'{case_folder}_{sub_file}.npy')
                np.save(npy_file_name, data_dict)

print("数据打包完成！")
