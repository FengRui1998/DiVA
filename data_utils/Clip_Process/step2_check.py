import os
import numpy as np
import moviepy.editor as mp
import librosa
import noisereduce as nr
from scipy.signal import butter, lfilter

# 带通滤波器设计
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

# 这是版本2
def check_silences(audio, sr, min_silence_duration=1.0, max_total_silence_duration=4.0, top_db=20,
                   min_gap_duration=1.0):
    # 使用 librosa.effects.split 检测非静音片段
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)

    # 初始化总静音时长
    total_silence_duration = 0.0
    non_silent_durations = []
    merged_intervals = []

    # 初始静音段：从音频开始到第一个非静音段开始的片段
    previous_end = 0

    # 合并相邻间隙小于 min_gap_duration 的非静音片段
    for i, (start, end) in enumerate(non_silent_intervals):
        if i == 0:
            merged_intervals.append([start, end])
        else:
            # 计算当前片段与上一个片段之间的间隙
            gap_duration = (start - merged_intervals[-1][1]) / sr
            if gap_duration < min_gap_duration:
                # 合并间隙小于指定时长的片段
                merged_intervals[-1][1] = end
            else:
                merged_intervals.append([start, end])

    # 计算总静音时长并记录非静音片段
    total_non_silent_duration = 0.0  # 初始化总非静音时长
    for idx, (start, end) in enumerate(merged_intervals):
        # 计算当前静音段的时长
        silence_duration = (start - previous_end) / sr

        # 如果静音时长大于 min_silence_duration，则累计到总静音时长
        if silence_duration >= min_silence_duration:
            total_silence_duration += silence_duration

        # 记录非静音片段的时长
        duration = (end - start) / sr
        non_silent_durations.append(duration)
        total_non_silent_duration += duration  # 累加到总非静音时长

        # 输出非静音片段的详细信息
        # print(
        #     f"Non-silent segment {idx + 1}: Start = {start / sr:.2f}s, End = {end / sr:.2f}s, Duration = {duration:.2f}s")

        # 更新 previous_end 为当前非静音段的结束位置
        previous_end = end

    # 处理最后一个非静音段到音频结束的静音段
    final_silence_duration = (len(audio) - previous_end) / sr
    if final_silence_duration >= min_silence_duration:
        total_silence_duration += final_silence_duration

    # 输出静音总时长   输出非静音总时长
    print(f'静音总时长： {total_silence_duration:.2f}s', f'非静音总时长： {total_non_silent_duration:.2f}s')

    # 判断总静音时长是否超过设定的阈值
    if total_silence_duration > max_total_silence_duration:
        return False, non_silent_durations, merged_intervals  # 返回无效标志和合并后的检测到的值
    else:
        return True, non_silent_durations, merged_intervals  # 返回有效标志和合并后的检测到的值


# 从视频中提取音频并直接处理
def process_video_audio(video_path, sr=22050):

    # 使用 moviepy 读取视频音频
    video = mp.VideoFileClip(video_path)

    audio = video.audio
    audio_array = np.array(audio.to_soundarray(fps=sr))


    # 将音频数据转换为单通道（如果是立体声）
    if audio_array.shape[1] > 1:
        audio_array = np.mean(audio_array, axis=1)

    # 保存降噪前的音频信号
    original_audio = audio_array.copy()

    # 后续处理步骤
    y = nr.reduce_noise(y=audio_array, sr=sr)
    y = bandpass_filter(y, lowcut=300, highcut=3400, fs=sr, order=6)
    # is_valid = check_silences(y, sr, min_non_silent_duration=3.0, top_db=20)
    is_valid, non_silent_durations, non_silent_intervals = check_silences(y, sr, min_silence_duration=1.0,
                                                                          max_total_silence_duration=3.0, top_db=20)

    # 释放资源
    video.reader.close()
    video.audio.reader.close_proc()

    return is_valid, y



# 处理单个视频文件
def process_video_file(video_path, sr=22050):
    is_valid, processed_audio = process_video_audio(video_path, sr=sr)
    if is_valid:
        print('   ')
        # print(f"The audio in {video_path} meets the silence criteria.")
    else:
        print(f"The audio in {video_path} does not meet the silence criteria and will be deleted.")
        os.remove(video_path)  # 删除不符合要求的视频文件


# 处理文件夹中的所有视频文件，包括子文件夹
def process_video_folder_recursive(folder_path, sr=22050):
    initial_count = 0
    final_count = 0

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.mp4'):
                initial_count += 1  # 统计初始文件数量
                video_path = os.path.join(root, filename)
                process_video_file(video_path, sr=sr)

    # 再次统计剩余的 .mp4 文件数量
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.mp4'):
                final_count += 1

    print(f"初始文件数量: {initial_count}")
    print(f"处理后文件数量: {final_count}")
    print(f"已删除文件数量: {initial_count - final_count}")

# 示例：处理文件夹中的所有视频文件
video_folder = r'G:\MCI_project\Dataset\tempp_1'  # 替换为你的文件夹路径
process_video_folder_recursive(video_folder)
