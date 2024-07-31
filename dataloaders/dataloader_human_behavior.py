import os
import numpy as np
from torch.utils.data import Dataset
from scipy.stats import zscore


class HumanBehaviorDataset(Dataset):
    def __init__(self, root_path, train=True):
        if train:
            partitions = range(0, 1)
        else:
            partitions = range(18, 20)

        all_data = []
        all_meta_info = []
        for partition_id in partitions:
            data_file_name = os.path.join(root_path, f"data_clips_partition{partition_id}_sample10000.pkl")
            meta_info_file_name = os.path.join(root_path, f"meta_info_partition{partition_id}_sample10000.pkl")

            data_clips = np.load(data_file_name, allow_pickle=True)
            meta_info = np.load(meta_info_file_name, allow_pickle=True)
            all_data.append(data_clips)
            all_meta_info.append(meta_info)

        self.data_video_clips = np.concatenate(all_data, axis=0)
        self.clip_meta_info = np.concatenate(all_meta_info, axis=0)
        print("Data shape:", self.data_video_clips.shape)
        print("Meta info shape:", self.clip_meta_info.shape)

    def __getitem__(self, index):
        clip = self.data_video_clips[index]
        clip_meta = self.clip_meta_info[index]
        return clip, clip_meta

    def __len__(self):
        return len(self.data_video_clips)


class HumanBehaviorDatasetInformer(Dataset):
    def __init__(self, args, flag, do_zscore=True):
        
        self.args = args
        self.do_zscore = do_zscore
        self.mean = None
        self.std = None

        self.data_video_clips, self.clip_meta_info = self.load_data(self.args.root_path, flag)
        self.data_pairs, self.pairs_meta_info = self.get_pairs(self.data_video_clips, self.clip_meta_info)

    def load_data(self, root_path, flag):
        if flag == "train":
            partitions = range(2, 12)
        elif flag == "test":
            partitions = range(0, 1)
        else:
            partitions = range(1, 2)
        all_data = []
        all_meta_info = []
        for partition_id in partitions:
            data_file_name = os.path.join(root_path, f"data_clips_partition{partition_id}_sample10000.pkl")
            meta_info_file_name = os.path.join(root_path, f"meta_info_partition{partition_id}_sample10000.pkl")

            data_clips = np.load(data_file_name, allow_pickle=True)
            meta_info = np.load(meta_info_file_name, allow_pickle=True)
            all_data.append(data_clips)
            all_meta_info.append(meta_info)

        data_video_clips = np.concatenate(all_data, axis=0)
        clip_meta_info = np.concatenate(all_meta_info, axis=0)

        data_video_clips = np.reshape(data_video_clips, (data_video_clips.shape[0], 140, self.args.seq_len))  # (sample_idx, feature_idx, time_idx)
        data_video_clips = np.transpose(data_video_clips, (0, 2, 1))

        if self.do_zscore:
            self.mean = data_video_clips.mean(axis=0)
            self.std = data_video_clips.std(axis=0)
            data_video_clips = zscore(data_video_clips, axis=0)

        print("Data shape:", data_video_clips.shape)
        print("Meta info shape:", clip_meta_info.shape)

        return data_video_clips, clip_meta_info

    def get_pairs(self, data, meta_info):
        pairs = []
        pairs_meta_info = []
        idx = 0
        while idx < len(data) - 1:
            # if two clips are from the same video
            if meta_info[idx][0] == meta_info[idx + 1][0]:
                concated = np.concatenate(
                    [data[idx], data[idx + 1, :self.args.pred_len]],
                    axis=0
                )
                pairs.append((data[idx], concated))
                pairs_meta_info.append((meta_info[idx], meta_info[idx + 1]))
            idx += 1
        return pairs, pairs_meta_info

    def __getitem__(self, index):
        clip_x, clip_y = self.data_pairs[index]
        return clip_x, clip_y

    def __len__(self):
        return len(self.data_pairs)
