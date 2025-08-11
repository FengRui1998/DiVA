import os
from sklearn.model_selection import KFold

# 输入根目录路径
root_folder_path = "/public/home/feng_rui/code/MCI_project/Dataset/Combine"

# 定义 Clip 和 Full 数据集根目录
root_folder_path_clip = os.path.join(root_folder_path, 'Clip')
root_folder_path_full = os.path.join(root_folder_path, 'Full')

# 定义类别文件夹
categories = ['AD', 'NC', 'MCI']

# 目标目录，用于保存生成的五折交叉验证文件
output_dir_clip = os.path.join(root_folder_path_clip, 'Cross_Validation_Splits')  # Clip 数据集结果保存目录
output_dir_full = os.path.join(root_folder_path_full, 'Cross_Validation_Splits')  # Full 数据集结果保存目录
os.makedirs(output_dir_clip, exist_ok=True)
os.makedirs(output_dir_full, exist_ok=True)

# 设置K折交叉验证参数
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 初始化存储每折训练和测试集的列表
fold_splits_clip = {i: {'train': [], 'test': []} for i in range(n_splits)}
fold_splits_full = {i: {'train': [], 'test': []} for i in range(n_splits)}

# 处理 Clip 和 Full 数据集
for dataset_dir, fold_splits, output_dir in [(root_folder_path_clip, fold_splits_clip, output_dir_clip),
                                             (root_folder_path_full, fold_splits_full, output_dir_full)]:
    # 遍历每个类别文件夹
    for category in categories:
        category_path = os.path.join(dataset_dir, category)

        # 获取所有病例子文件夹
        patient_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

        # 进行K折划分
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(patient_folders)):
            train_folders = []
            test_folders = []

            # 进入每个训练集的病例文件夹
            for i in train_idx:
                folder_path = os.path.join(category_path, patient_folders[i])
                # 获取内部的三个子文件夹路径，并添加到 train_folders 列表
                subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if
                              os.path.isdir(os.path.join(folder_path, subfolder))]
                train_folders.extend(subfolders)

            # 进入每个测试集的病例文件夹
            for i in test_idx:
                folder_path = os.path.join(category_path, patient_folders[i])
                # 获取内部的三个子文件夹路径，并添加到 test_folders 列表
                subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if
                              os.path.isdir(os.path.join(folder_path, subfolder))]
                test_folders.extend(subfolders)

            # 将每个类别的划分结果添加到 fold_splits 中
            fold_splits[fold_idx]['train'].extend(train_folders)
            fold_splits[fold_idx]['test'].extend(test_folders)

# 生成最终的训练集和测试集文件
for dataset_name, fold_splits, output_dir in [('Clip', fold_splits_clip, output_dir_clip),
                                              ('Full', fold_splits_full, output_dir_full)]:
    for fold_idx in range(n_splits):
        train_file_path = os.path.join(output_dir, f"MCI_set_{fold_idx + 1}_train.txt")
        test_file_path = os.path.join(output_dir, f"MCI_set_{fold_idx + 1}_test.txt")

        # 写入训练集文件
        with open(train_file_path, "w") as train_file:
            train_file.write("\n".join(fold_splits[fold_idx]['train']) + "\n")

        # 写入测试集文件
        with open(test_file_path, "w") as test_file:
            test_file.write("\n".join(fold_splits[fold_idx]['test']) + "\n")

        # 输出当前折的信息
        print(f"{dataset_name} Fold {fold_idx + 1}")
        print(f"Train File: {train_file_path}")
        print(f"Test File: {test_file_path}")
        print("-----")
