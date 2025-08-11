'''

该代码的作用是将数据集里中文文件名替换成拼音首字母缩写

'''
import glob
import os
import shutil
import pandas as pd
from pypinyin import lazy_pinyin, Style


def get_pinyin_initials(chinese_name):
    # 获取中文名称的拼音首字母
    initials = lazy_pinyin(chinese_name, style=Style.FIRST_LETTER)
    return ''.join(initials)


def check_moca_column(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path, encoding='latin1')
        if 'MoCA' not in df.columns:
            print(f"'MoCA' column not found in: {csv_file_path}")
        elif df['MoCA'].isnull().all():
            print(f"'MoCA' column has no valid values in: {csv_file_path}")
        # else:
            # print(f"'MoCA' column found with valid values in: {csv_file_path} - Success")
    except Exception as e:
        print(f"Error reading {csv_file_path}: {e}")

def rename_folders_and_check_csv(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            parts = folder_name.split('_')
            if len(parts) == 3:
                chinese_name = parts[0]
                pinyin_initials = get_pinyin_initials(chinese_name)
                new_folder_name = f"{pinyin_initials}_{parts[1]}_{parts[2]}"
                new_folder_path = os.path.join(root_dir, new_folder_name)

                # 检查文件夹内的csv文件
                csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
                if csv_files:
                    for csv_file in csv_files:
                        check_moca_column(csv_file)
                    try:
                        os.rename(folder_path, new_folder_path)
                        # print(f"Renamed: {folder_path} -> {new_folder_path}")
                    except Exception as e:
                        print(f"Error renaming {folder_path}: {e}")
                else:
                    print(f"CSV file not found in: {folder_path}")


if __name__ == "__main__":
    # 指定包含AD、HC、MCI文件夹的根目录路径
    main_directory = r'G:\MCI_project\data_utils\Data_2'

    # 在主目录中寻找AD、HC、MCI文件夹并重命名其子文件夹
    sub_directories = ["AD", "NC", "MCI"]
    for sub_dir in sub_directories:
        root_dir = os.path.join(main_directory, sub_dir)
        if os.path.exists(root_dir):
            rename_folders_and_check_csv(root_dir)
        else:
            print(f"Directory not found: {root_dir}")
