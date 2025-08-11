import glob
import os
import pandas as pd

def check_moca_column(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path, encoding='latin1')
        if 'MoCA' not in df.columns or df['MoCA'].isnull().all():
            print(f"'MoCA' column missing or has no valid values in: {csv_file_path}")
        # 如果存在且有有效值，则不打印任何信息
    except Exception as e:
        print(f"Error reading {csv_file_path}: {e}")

def check_csv_files_in_folders(root_dir):
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            if csv_files:
                for csv_file in csv_files:
                    check_moca_column(csv_file)
            else:
                print(f"No CSV files found in: {folder_path}")

if __name__ == "__main__":
    main_directory = r'G:\MCI_project\data_utils\Data_2'

    sub_directories = ["AD", "NC", "MCI"]
    for sub_dir in sub_directories:
        root_dir = os.path.join(main_directory, sub_dir)
        if os.path.exists(root_dir):
            check_csv_files_in_folders(root_dir)
        else:
            print(f"Directory not found: {root_dir}")
