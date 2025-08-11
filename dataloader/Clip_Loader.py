import os.path
import librosa
import pandas as pd
from torch.utils import data
import glob
import os
from data_loader.video_transform import *
import torchaudio
from data_loader.audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask
class VideoRecord(object):
    """
    封装病例文件夹路径、帧数、标签和附加信息的类
    """
    def __init__(self, path, num_frames, label, info=None):
        self._path = path        # 病例文件夹路径
        self._num_frames = num_frames
        self._label = label      # MoCA生成的标签
        self._info = info        # 额外的病人信息，如年龄、学历、MMSE等

    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def label(self):
        return self._label

    @property
    def info(self):
        return self._info  # 返回额外的病人信息


class VideoDataset(data.Dataset):
    def __init__(self, list_file, mode, video_transform,audio_transform, image_size, num_class):
        self.list_file = list_file
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.image_size = image_size
        self.num_class = num_class
        self.mode = mode
        self._parse_list()  # 调用解析函数，解析list_file

    def _parse_list(self):
        """
        解析list_file，提取病例文件夹路径，并从frames_face中计算帧数，从label.csv中读取标签。
        """
        self.video_list = []

        # 读取list_file文件
        with open(self.list_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # 获取病例文件夹路径
            case_folder_path = line.strip()

            # 获取frames_face文件夹中的所有帧(.jpg文件)
            # frames_face_folder = os.path.join(case_folder_path, 'frames_face')
            num_frames = len(glob.glob(os.path.join(case_folder_path, '*.jpg')))

            # Person_Data.csv文件中读取标签
            label_file = os.path.join(case_folder_path, 'Person_Data.csv')

            # 尝试读取CSV文件并处理编码问题
            try:
                df = pd.read_csv(label_file, encoding='gbk')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(label_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(label_file, encoding='latin1')

            # 处理年龄字段
            age_data = torch.tensor(df['年龄'].values, dtype=torch.int64)

            # 将学历数据映射为数值
            edu_mapping = {'小学': 0, '初中': 1, '高中': 2, '大专及以上': 3}  # 根据需要修改映射
            edu_data = torch.tensor([edu_mapping.get(edu, -1) for edu in df['学历'].values], dtype=torch.int64)

            # 处理MMSE字段，填充缺失值为-1
            if 'MMSE' in df.columns and not df['MMSE'].isnull().all():
                mmse_data = torch.tensor(df['MMSE'].fillna(-1).values, dtype=torch.int64)
            else:
                mmse_data = torch.tensor([-1], dtype=torch.int64)

            # 处理MoCA字段
            moca_data = torch.tensor(df['MoCA'].values, dtype=torch.int64)

            MoCA_value = moca_data.item()  # 获取 MoCA 值，假设每个病例有一个 MoCA 值

            # 根据 num_class 来确定标签的生成规则
            if self.num_class == 3:
                if MoCA_value > 25:
                    label = 0  # 正常人
                elif 18 < MoCA_value <= 25:
                    label = 1  # MCI患者
                else:
                    label = 2  # AD患者

            elif self.num_class == 2:
                if MoCA_value > 25:
                    label = 0  # 正常人
                else:
                    label = 1  # 不正常（包括MCI和AD）

            elif self.num_class ==4:
                if MoCA_value > 25:
                    label = 0 # 正常人
                elif 18 < MoCA_value <= 25:
                    label = 1 #轻度损伤
                elif 9 < MoCA_value <= 18:
                    label = 2 #中度损伤
                else:
                    label = 3 #重度损伤


            # 额外的信息，包含年龄、学历、MMSE、MoCA等信息
            info = {
                'age': age_data,
                'education': edu_data,
                'mmse': mmse_data,
                'moca': moca_data
            }

            # 创建VideoRecord实例，将label和info一起传递
            record = VideoRecord(path=case_folder_path, num_frames=num_frames, label=label, info=info)
            self.video_list.append(record)

        print(f'Number of {self.mode} files: {len(self.video_list)}')


    def _wav2fbank(self, filename, target_length=1024):
        """
        从音频文件中提取 mel-spectrogram 特征，支持训练和测试模式。
        训练模式随机截取片段，测试模式从中间截取。

        Args:
        - filename: 音频文件的路径
        - segment_length_seconds: 每个片段的秒数
        - target_length: 最终 Mel-spectrogram 的帧数
        - mode: 'train' 或 'test'，决定是随机截取还是中间截取
        """
        waveform, sr = torchaudio.load(filename)
        waveform = waveform.squeeze().numpy()

        waveform = self.audio_transform(samples=waveform, sample_rate=sr)

        # waveform = waveform - waveform.mean()

        # 转换增强后的音频为 torch.Tensor，以便后续使用 torchaudio 进行 Mel-spectrogram 计算
        waveform = torch.tensor(waveform).float()

        # 确保 audio 数据的 shape 是 (num_channels, num_samples)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)  # 如果是单通道，添加一个通道维度
        # 计算 Mel-spectrogram
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                                  use_energy=False, window_type='hanning',
                                                  num_mel_bins=128, dither=0.0, frame_shift=10)

        # 填充或截断 Mel-spectrogram
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        elif p < 0:
            fbank = fbank[:target_length, :]

        return fbank, 0

    def get(self, record):
        """
        根据给定的索引提取frames_face文件夹中的视频帧，并提取audio.wav中的音频特征。
        """
        # 获取frames_face文件夹中的所有帧路径
        video_frames_path = sorted(glob.glob(os.path.join(record.path, '*.jpg')))

        images = []
        # 遍历所有图像文件，逐帧加载
        for img_path in video_frames_path:
            img = Image.open(img_path).convert('RGB')  # 读取并转换为 RGB
            images.append(img)


        # 获取病例文件夹中的audio.wav路径
        audio_path = os.path.join(record.path, 'audio.wav')
        fbank, _ = self._wav2fbank(audio_path)

        # 图像转换处理
        images = self.video_transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        # 返回图像和音频特征
        return images, fbank.unsqueeze(0), record.label#record.info['moca']

    def __getitem__(self, index):
        record = self.video_list[index]
        return self.get(record)

    def __len__(self):
        return len(self.video_list)

sometimes = lambda aug: Sometimes(0.5, aug)

def train_clip_data_loader(list_file, image_size, num_class, augmentation=False):

    if augmentation:
        print('use_augmentation')
        # 加数据增强
        video_transform = torchvision.transforms.Compose([
            GroupResize(image_size),
            sometimes(HorizontalFlip()),
            sometimes(ColorJitter(brightness=0.5)),
            sometimes(RandomRotation(15)),
            # sometimes(GaussianBlur(2)),
            # sometimes(RandomShear(0.1, 0.1)),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        audio_transform = Compose([
                                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                                    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                                    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                                    Shift(min_shift=0.2, max_shift=0.2, p=0.5),
                                    TimeMask(min_band_part=0.1, max_band_part=0.3, fade=True, p=0.5)
                                ])

    else:
        # 不加数据增强
        video_transform = torchvision.transforms.Compose([
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        audio_transform = Compose([
                                ])

    train_data = VideoDataset(list_file=list_file,
                              mode='train',
                              video_transform=video_transform,
                              audio_transform =audio_transform,
                              image_size=image_size,
                              num_class=num_class)

    return train_data


def test_clip_data_loader(list_file, image_size, num_class):
    video_transform = torchvision.transforms.Compose([
        GroupResize(image_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    audio_transform = Compose([])

    test_data = VideoDataset(list_file=list_file,
                             mode='test',
                             video_transform=video_transform,
                             audio_transform=audio_transform,
                             image_size=image_size,
                             num_class=num_class)
    return test_data

if __name__ == '__main__':
    # 创建数据加载器
    train_list = '/public/home/feng_rui/code/MCI_project/Dataset/Data_Clip/Cross_Validation_Splits/MCI_set_1_train.txt'
    test_list = '/public/home/feng_rui/code/MCI_project/Dataset/Data_Clip/Cross_Validation_Splits/MCI_set_1_test.txt'

    train_data = train_clip_data_loader(list_file=train_list,
                                   image_size=224,
                                   num_class=2,
                                   augmentation=False)

    test_data = test_clip_data_loader(list_file=test_list,
                                 image_size=224,
                                 num_class=2)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=2,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)

    # 遍历 train_loader，并打印每个批次的数据
    for i, (images, fbank, labels) in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        # 打印图像信息
        print(f"  Images shape: {images.shape}")
        # print(images)
        # 打印标签信息
        print(f"  Labels: {labels}")
        # 打印音频特征信息
        print(f"  Fbank shape: {fbank.shape}")
        # print(fbank)
        # 如果你只想打印前几个批次，可以设置一个限制
        if i == 0:  # 打印前 3 个批次
            break

    print('---------------------------------------------------')

    for i, (images, fbank, labels) in enumerate(val_loader):
        print(f"Batch {i + 1}:")
        # 打印图像信息
        print(f"  Images shape: {images.shape}")
        # print(images)
        # 打印标签信息
        print(f"  Labels: {labels}")
        labels2 = labels.float().unsqueeze(1)
        print(labels2)
        print(labels2.shape)
        # 打印音频特征信息
        print(f"  Fbank shape: {fbank.shape}")
        # print(fbank)
        # 如果你只想打印前几个批次，可以设置一个限制
        if i == 0:  # 打印前 3 个批次
            break

