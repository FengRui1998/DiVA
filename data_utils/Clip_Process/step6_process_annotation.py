import os
from sklearn.model_selection import KFold

# 定义数据集的根目录
root_folder_path = r"G:\MCI_project\Dataset\data_process\DATA_Clip_split_check_face"  # 数据集根目录

# 定义类别文件夹
categories = ['AD', 'NC', 'MCI']

# 目标目录，用于保存生成的五折交叉验证文件
output_dir = os.path.join(root_folder_path, 'Cross_Validation_Splits') # 结果保存目录
os.makedirs(output_dir, exist_ok=True)

# 设置K折交叉验证参数
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 初始化存储每折训练和测试集的列表
fold_splits = {i: {'train': [], 'test': []} for i in range(n_splits)}

# 遍历每个类别文件夹
for category in categories:
    category_path = os.path.join(root_folder_path, category)

    # 获取所有病例子文件夹
    patient_folders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

    # 进行K折划分
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(patient_folders)):
        # 获取当前折的训练集和测试集
        train_folders = [patient_folders[i] for i in train_idx]
        test_folders = [patient_folders[i] for i in test_idx]

        # 将每个类别的划分结果添加到 fold_splits 中
        for folder in train_folders:
            npy_files = [os.path.join(category_path, folder, f) for f in os.listdir(os.path.join(category_path, folder))
                         if f.endswith('.npy')]
            fold_splits[fold_idx]['train'].extend(npy_files)

        for folder in test_folders:
            npy_files = [os.path.join(category_path, folder, f) for f in os.listdir(os.path.join(category_path, folder))
                         if f.endswith('.npy')]
            fold_splits[fold_idx]['test'].extend(npy_files)

# 生成最终的训练集和测试集文件
for fold_idx in range(n_splits):
    train_file_path = os.path.join(output_dir, f"MCI_set_{fold_idx + 1}_train.txt")
    test_file_path = os.path.join(output_dir, f"MCI_set_{fold_idx + 1}_test.txt")

    with open(train_file_path, "w") as train_file:
        train_file.write("\n".join(fold_splits[fold_idx]['train']) + "\n")

    with open(test_file_path, "w") as test_file:
        test_file.write("\n".join(fold_splits[fold_idx]['test']) + "\n")

    # 输出当前折的信息
    print(f"Fold {fold_idx + 1}")
    print(f"Train File: {train_file_path}")
    print(f"Test File: {test_file_path}")
    print("-----")
