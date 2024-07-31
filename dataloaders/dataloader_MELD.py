import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MELDLatentRepresentation(Dataset):
    def __init__(self, data_path, split, task):
        latent_representation, meta_info = self.load_data(data_path, split)

        self.emotion, self.sentiment, self.idx_to_emotion, self.idx_to_sentiment, self.emotion_to_idx, self.sentiment_to_idx = self.unpack_meta_info(meta_info)

        self.latent_representation = latent_representation
        if task == "emotion":
            self.label = self.emotion
        elif task == "sentiment":
            self.label = self.sentiment
        else:
            raise ValueError(f"Unknown task: {task}")

        assert len(self.latent_representation) == len(self.label)

    def load_data(self, root_path, split):
        latent_representation_file_name = os.path.join(root_path, f"latent_representation_{split}.pkl")
        meta_info_file_name = os.path.join(root_path, f"meta_info_{split}.npy")

        latent_representation = np.load(latent_representation_file_name, allow_pickle=True)
        meta_info = np.load(meta_info_file_name, allow_pickle=True)

        return latent_representation, meta_info

    def unpack_meta_info(self, meta_info):
        emotion = meta_info[:, 2]
        sentiment = meta_info[:, 3]

        emotion_to_idx = {}
        idx_to_emotion = {}
        for i, e in enumerate(np.unique(emotion)):
            emotion_to_idx[e] = i
            idx_to_emotion[i] = e
        emotion = torch.tensor([emotion_to_idx[e] for e in emotion])

        sentiment_to_idx = {}
        idx_to_sentiment = {}
        for i, s in enumerate(np.unique(sentiment)):
            sentiment_to_idx[s] = i
            idx_to_sentiment[i] = s
        sentiment = torch.tensor([sentiment_to_idx[s] for s in sentiment])

        return emotion, sentiment, idx_to_emotion, idx_to_sentiment, emotion_to_idx, sentiment_to_idx

    def __getitem__(self, index):
        return self.latent_representation[index], self.label[index]

    def __len__(self):
        return len(self.latent_representation)
