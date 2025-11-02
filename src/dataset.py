from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from augmentations import *
import numpy as np
import scipy
import torch


class EMGDataset(Dataset):
    def __init__(self, config: dict[str, any]):
        self.input_directory = config["input_directory"]
        self.batch_size = config["batch_size"]

        self.repetitions = config["repetitions"]
        self.n_reps = len(self.repetitions)

        self.dim = config["dim"]
        self.classes = sorted(config["classes"])
        self.class_index = {cls: i for i, cls in enumerate(self.classes)}

        self.pad_value = config["pad_value"]
        self.pad_len = config["pad_len"]

        self.window_size = config["window_size"]
        self.window_step = config["window_step"]
        self.size_factor = config["size_factor"]

        self.permutation = config["permutation"]
        self.rotation = config["rotation"]
        self.rotation_mask = config["rotation_mask"]
        self.time_warping = config["time_warping"]
        self.scale_sigma = config["scale_sigma"]
        self.mag_warping = config["mag_warping"]
        self.noise_snr_db = config["noise_snr_db"]

        self.sample_weight = config["sample_weight"]
        
        self.data_type = "rms"

        self.__load_data()
        self.__validate_params()
        self.__generate()

    def __load_data(self):
        emg_signals, labels, repetitions = [], [], []
        self.max_sequence_length = 0

        if 0 in self.classes:
            rest_repetition_groups = list(
                zip(
                    np.random.choice(self.repetitions, (self.n_reps), replace=False),
                    np.random.choice(
                        [
                            gesture_class
                            for gesture_class in self.classes
                            if gesture_class != 0
                        ],
                        (self.n_reps),
                        replace=False,
                    ),
                )
            )

        for directory_index in tqdm(range(len(self.input_directory)), desc="Loading data"):
            for gesture_class in [
                gesture_class for gesture_class in self.classes if gesture_class != 0
            ]:
                for repetition in self.repetitions:
                    file_path = "{}/gesture-{}/{}/rep-{:02d}.mat".format(
                        self.input_directory[directory_index],
                        int(gesture_class),
                        self.data_type,
                        int(repetition),
                    )

                    data = scipy.io.loadmat(file_path)
                    emg_signal = data["emg"].copy()

                    emg_signal = low_pass_filter(emg_signal, sampling_freq=100)

                    if len(emg_signal) > self.max_sequence_length:
                        self.max_sequence_length = len(emg_signal)

                    emg_signals.append(emg_signal)
                    labels.append(int(np.squeeze(data["stimulus"])[0]))
                    repetitions.append(int(np.squeeze(data["repetition"])[0]))

            if 0 in self.classes:
                for repetition, gesture_class in rest_repetition_groups:
                    file_path = "{}/gesture-0/{}/rep-{:02d}_{:02d}.mat".format(
                        self.input_directory[directory_index],
                        self.data_type,
                        int(repetition),
                        int(gesture_class),
                    )

                    data = scipy.io.loadmat(file_path)
                    emg_signal = data["emg"].copy()

                    emg_signal = low_pass_filter(emg_signal, sampling_freq=100)

                    if len(emg_signal) > self.max_sequence_length:
                        self.max_sequence_length = len(emg_signal)

                    emg_signals.append(emg_signal)
                    labels.append(int(np.squeeze(data["stimulus"])[0]))
                    repetitions.append(int(np.squeeze(data["repetition"])[0]))

        self.emg_signals = emg_signals
        self.labels = labels
        self.repetitions = repetitions

    def __validate_params(self):
        if ((self.dim[0] is None) or (self.pad_len is None)) and (
            (self.window_size == 0) or (self.window_step == 0)
        ):
            self.dim = (self.max_sequence_length, *self.dim[1:])
            self.pad_len = self.max_sequence_length
            self.window_step = 0
            self.window_size = 0
    
    def __augment(self):
        self.emg_signals_aug, self.labels_aug, self.repetitions_aug = [], [], []

        emg_np = np.array(self.emg_signals, dtype="object")
        labels_np = np.array(self.labels)
        reps_np = np.array(self.repetitions)

        for i in tqdm(range(emg_np.shape[0]), desc="Augmenting data"):
            for _ in range(self.size_factor):
                x = emg_np[i]

                if self.permutation != 0:
                    x = permute(x, nPerm=self.permutation)

                if self.rotation != 0:
                    x = rotate(x, rotation=self.rotation, mask=self.rotation_mask)

                if self.time_warping != 0:
                    x = time_warp(x, sigma=self.time_warping)

                if self.scale_sigma != 0:
                    x = scale(x, sigma=self.scale_sigma)

                if self.mag_warping != 0:
                    x = mag_warp(x, sigma=self.mag_warping)

                if self.noise_snr_db != 0:
                    x = jitter(x, snr_db=self.noise_snr_db)

                if (
                    self.permutation
                    or self.rotation
                    or self.time_warping
                    or self.scale_sigma
                    or self.mag_warping
                    or self.noise_snr_db
                ):
                    self.emg_signals_aug.append(torch.tensor(x, dtype=torch.float32))
                    self.labels_aug.append(labels_np[i])
                    self.repetitions_aug.append(reps_np[i])

            self.emg_signals_aug.append(torch.tensor(self.emg_signals[i].copy(), dtype=torch.float32))
            self.labels_aug.append(self.labels[i])
            self.repetitions_aug.append(self.repetitions[i])
    
        self.labels_aug = torch.tensor(self.labels_aug, dtype=torch.long)
        self.repetitions_aug = torch.tensor(self.repetitions_aug, dtype=torch.long)

    def __make_segments(self):
        x_offsets = []

        num_samples = len(self.emg_signals_aug)
        lengths = [x.shape[0] for x in self.emg_signals_aug]

        if self.window_size != 0:
            for i in tqdm(range(num_samples), desc="Creating segments"):
                signal_len = lengths[i]
                if signal_len <= self.window_size:
                    continue  # skip if signal too short
                max_start = signal_len - self.window_size
                for j in range(0, max_start, self.window_step):
                    x_offsets.append((i, j))
        else:
            # no segmentation, treat entire sequence as one window
            x_offsets = [(i, 0) for i in range(num_samples)]

        self.x_offsets = x_offsets

    def __make_sample_weights(self):
        self.class_weights = np.zeros(len(self.classes))

        for index in self.indexes:
            i, j = self.x_offsets[index]
            self.class_weights[self.class_index[int(self.labels_aug[i])]] += 1

        self.class_weights = 1 / self.class_weights
        self.class_weights /= np.max(self.class_weights)

    def __generate(self):
        self.__augment()
        self.__make_segments()

        self.indexes = np.arange(len(self.x_offsets))
        if self.batch_size > len(self.x_offsets):
            self.batch_size = len(self.x_offsets)

        self.class_weights = []
        if self.sample_weight:
            self.__make_sample_weights()

        if (self.window_size == 0) and (self.pad_len is not None):
            self.emg_signals_aug = pad_sequence(
                self.emg_signals_aug, padding_value=self.pad_value, batch_first=True
            )

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __data_generation(self, indexes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X = torch.zeros(
            (self.batch_size, *self.dim), dtype=torch.float32, device=device
        )
        y = torch.zeros((self.batch_size,), dtype=torch.long, device=device)

        if self.sample_weight:
            w = torch.zeros((self.batch_size,), dtype=torch.float32, device=device)

        for k, index in enumerate(indexes):
            i, j = self.x_offsets[index]

            if self.window_size != 0:
                x_aug = self.emg_signals_aug[i, j : j + self.window_size].clone()
            else:
                x_aug = self.emg_signals_aug[i].clone()

            if getattr(self, "min_max_norm", False):
                mask = x_aug != self.pad_value
                if mask.any():
                    x_valid = x_aug[mask]
                    min_x, max_x = x_valid.min(), x_valid.max()
                    x_aug[mask] = (x_aug[mask] - min_x) / (max_x - min_x)

            if torch.prod(torch.tensor(x_aug.shape)) == torch.prod(
                torch.tensor(self.dim)
            ):
                x_aug = x_aug.view(self.dim)
            else:
                raise ValueError(
                    f"Generated sample dimension mismatch. Found {tuple(x_aug.shape)}, expected {self.dim}."
                )

            X[k] = x_aug
            y[k] = self.class_index[int(self.labels_aug[i])]

            if self.sample_weight:
                w[k] = self.class_weights[y[k]]

        if hasattr(self, "n_classes"):
            n_classes = self.n_classes
        else:
            n_classes = len(self.classes)

        y_onehot = torch.nn.functional.one_hot(y, num_classes=n_classes).float()

        if self.sample_weight:
            return X, y_onehot, w
        else:
            return X, y_onehot

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        output = self.__data_generation(indexes)

        return output


if __name__ == "__main__":
    config = {
        "repetitions": [1, 3, 4, 6, 8, 9, 10],
        "input_directory": [f"data/processed/subject-{i:02d}" for i in range(1, 28)],
        "batch_size": 128,
        "sample_weight": True,
        "dim": [1, 10],
        "classes": [i for i in range(53)],
        "shuffle": True,
        "noise_snr_db": 25,
        "scale_sigma": 0.0,
        "window_size": 0,
        "window_step": 0,
        "rotation": 0,
        "rotation_mask": None,
        "time_warping": 0.2,
        "mag_warping": 0.2,
        "permutation": 0,
        "size_factor": 10,
        "pad_len": None,
        "pad_value": -10.0,
        "min_max_norm": False,
        "update_after_epoch": False,
    }
    
    dataset = EMGDataset(config)
    print(len(dataset))
    
    print(dataset.__getitem__(0))
