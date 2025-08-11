import os
import shutil
from moviepy.editor import VideoFileClip


def process_all_videos(base_dir, output_base_dir, window_size=6, step_size=3, target_fps=30):
    all_logs = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mp4'):
                input_video_path = os.path.join(root, file)

                # 获取相对于base_dir的相对路径，并构建输出路径
                relative_path = os.path.relpath(root, base_dir)
                output_folder = os.path.join(output_base_dir, relative_path)

                # 创建输出目录（如果不存在）
                os.makedirs(output_folder, exist_ok=True)

                # 加载视频
                video = VideoFileClip(input_video_path)
                audio = video.audio
                video_duration = video.duration

                # 计算切片参数
                segment_duration = window_size
                segment_overlap = step_size / window_size
                num_segments = int((video_duration / (segment_duration * (1 - segment_overlap))) - 1)

                for s in range(num_segments):
                    start_time = s * segment_duration * (1 - segment_overlap)
                    end_time = start_time + segment_duration

                    # 提取视频片段
                    video_segment = video.subclip(start_time, end_time)

                    # 定义输出文件名
                    base_filename = os.path.splitext(file)[0]
                    video_filename = os.path.join(output_folder, f"{base_filename}_segment_{s + 1:03d}_video.mp4")
                    # audio_filename = os.path.join(output_folder, f"{base_filename}_segment_{s + 1}_audio.wav")

                    # 保存视频片段
                    try:
                        video_segment.write_videofile(video_filename, codec="libx264", audio_codec="aac",
                                                      fps=target_fps)
                    except Exception as e:
                        print(f"Error saving video segment {s + 1}: {e}")

                video.close()  # 处理完视频后，关闭主视频对象

            # 检查并复制CSV文件
            elif file.endswith('.csv'):
                input_csv_path = os.path.join(root, file)

                # 获取相对于base_dir的相对路径，并构建输出路径
                relative_path = os.path.relpath(root, base_dir)
                output_folder = os.path.join(output_base_dir, relative_path)

                # 创建输出目录（如果不存在）
                os.makedirs(output_folder, exist_ok=True)

                # 定义输出CSV文件路径
                output_csv_path = os.path.join(output_folder, file)

                # 复制CSV文件
                shutil.copyfile(input_csv_path, output_csv_path)
                print(f"Copied CSV file to {output_csv_path}")


# 使用示例
base_dir = r"G:\MCI_project\Dataset\tempp_1"  # 原始视频文件目录
output_base_dir = r"G:\MCI_project\Dataset\DATA_Cilp_split"  # 输出基准目录
process_all_videos(base_dir, output_base_dir, window_size=6, step_size=3)
