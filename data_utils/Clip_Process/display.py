import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置要读取的文件路径
file_path = r'G:\MCI_project\Dataset\DATA_Clip_split_check_face_npy\AD\dch_72_1716708152447\dch_72_1716708152447_VID_21716708240375_segment_005.npy'

# 读取npy文件
data = np.load(file_path, allow_pickle=True).item()  # 使用 item() 以访问字典

# 访问数据
images = data['images']  # 获取图像数据
audio = data['audio_data']  # 获取音频文件路径
sample_rate = data['sample_rate']
age = data['age']  # 获取年龄
education = data['education']  # 获取学历
mmse = data['mmse']  # 获取MMSE
moca = data['moca']  # 获取MoCA
facial_landmarks = data['facial_landmarks']  # 获取面部关键点数据

# 打印其他相关信息

print(f'Age: {age}, Education: {education}, MMSE: {mmse}, MoCA: {moca}')
print(f'facial_landmarks: {facial_landmarks.shape}')
# 显示第一张图像
if images:
    image_rgb = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.title(f'Original Image\nAge: {age}, Education: {education}, MMSE: {mmse}, MoCA: {moca}')
    plt.axis('off')
    plt.show()
else:
    print("没有找到图像数据。")

# 可视化 landmark 点
if len(facial_landmarks) > 0:
    # 获取 X 和 Y 坐标，假设是标准化的（0-1），根据图像大小缩放
    img_height, img_width, _ = image_rgb.shape
    x_landmarks = facial_landmarks[0, :68] * img_width
    y_landmarks = facial_landmarks[0, 68:136] * img_height

    # 创建一个空白背景用于展示 landmarks
    blank_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # 白色背景

    plt.figure(figsize=(8, 8))
    plt.imshow(blank_image)
    plt.scatter(x_landmarks, y_landmarks, c='red', s=10)  # 使用红色标记关键点
    plt.title("Facial Landmarks")
    plt.axis('off')
    plt.show()
else:
    print("没有找到面部关键点数据。")


# 可视化音频波形
if audio is not None:
    plt.figure(figsize=(12, 4))
    time_axis = np.linspace(0, len(audio) / sample_rate, num=len(audio))
    plt.plot(time_axis, audio, color='blue')
    plt.title('Audio Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
else:
    print("没有找到音频数据，无法进行可视化。")
print(f'audio: {audio.shape}')