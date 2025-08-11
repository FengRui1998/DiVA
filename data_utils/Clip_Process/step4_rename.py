import os
import glob

from tqdm import tqdm

# 设置根目录
root_dir = r'G:\MCI_project\Dataset\DATA_Clip_split_check_face'

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
                # 获取csv和wav文件
                csv_files = glob.glob(os.path.join(sub_folder_path, '*.csv'))
                wav_files = glob.glob(os.path.join(sub_folder_path, '*.wav'))
                jpg_files = glob.glob(os.path.join(sub_folder_path, '*.jpg'))

                # 重命名CSV文件
                if len(csv_files) >= 2:
                    os.rename(csv_files[0], os.path.join(sub_folder_path, 'Person_Data.csv'))
                    os.rename(csv_files[1], os.path.join(sub_folder_path, 'Openface_Label.csv'))

                # 重命名WAV文件
                if wav_files:
                    os.rename(wav_files[0], os.path.join(sub_folder_path, 'audio.wav'))

                # 重命名JPG文件
                for idx, jpg_file in enumerate(sorted(jpg_files), start=1):
                    new_name = os.path.join(sub_folder_path, f'frame_{idx:03}.jpg')
                    os.rename(jpg_file, new_name)

print("重命名完成！")
