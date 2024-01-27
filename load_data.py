import os
import glob
from torch.utils.data import Dataset, DataLoader, Subset
import feature_extraction
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted(os.listdir(folder_path))
        self.speakers = sorted(set(record.split('-')[0] for record in self.files))
        self.data = self.load_data()

    def load_data(self):
        data = []
        for recording in self.files:
            path = os.path.join(self.folder_path, recording)
            label = recording.split('-')[0]
            data.append((feature_extraction.get_features(path), label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor = self.data[index]

        # Sample positive from the same class
        positive_class_samples = [i for i, (features, label) in enumerate(self.data) if label == anchor[1]]
        positive_index = np.random.choice(positive_class_samples)

        # Sample negative from a different class
        negative_class_samples = [i for i, (features, label) in enumerate(self.data) if label != anchor[1]]
        negative_index = np.random.choice(negative_class_samples)

        return index, positive_index, negative_index