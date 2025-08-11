import os

def get_filenames(folder_path):
    """
    获取指定文件夹中的所有文件名（不包括路径）
    """
    return set(os.listdir(folder_path))

def compare_folders(folder1, folder2):
    """
    比较两个文件夹内的文件名，并找出不一致的部分
    """
    # 获取两个文件夹中的文件名集合
    files1 = get_filenames(folder1)
    files2 = get_filenames(folder2)

    # 计算在 folder1 中但不在 folder2 中的文件
    only_in_folder1 = files1 - files2
    # 计算在 folder2 中但不在 folder1 中的文件
    only_in_folder2 = files2 - files1

    # 输出不一致的文件
    if only_in_folder1:
        print(f"在 {folder1} 中，但不在 {folder2} 中的文件:")
        for file in only_in_folder1:
            print(file)
    else:
        print(f"{folder1} 中的所有文件都存在于 {folder2} 中。")

    if only_in_folder2:
        print(f"\n在 {folder2} 中，但不在 {folder1} 中的文件:")
        for file in only_in_folder2:
            print(file)
    else:
        print(f"{folder2} 中的所有文件都存在于 {folder1} 中。")

# 设置两个文件夹的路径
folder1 = r'G:\MCI_project\Dataset\DATA_Clip_split_check\NC'
folder2 = r'G:\MCI_project\Dataset\DATA_Clip_split_check_face\NC'

# 比较两个文件夹内的文件名
compare_folders(folder1, folder2)
