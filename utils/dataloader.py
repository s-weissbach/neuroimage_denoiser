from utils.trainfiles import trainfiles
import numpy as np
import tifffile
import torch


class dataloader:
    def __init__(
        self,
        t_files: trainfiles,
        batch_size: int,
        sequence_len: int,
        train_height: int,
        train_width: int,
        load_multiple_targets_per_file: bool = False,
        n_multiple_targets: int = 5,
        max_intensity: int = 65535,
    ):
        if load_multiple_targets_per_file and n_multiple_targets > batch_size:
            raise ValueError(
                f"n_multiple_targets ({n_multiple_targets}) must be smaller or equal to epoch_size ({batch_size})."
            )
        elif load_multiple_targets_per_file and batch_size % n_multiple_targets != 0:
            raise ValueError(
                f"epoch_size ({batch_size}) must be divisible by n_multiple_targets ({n_multiple_targets})."
            )
        np.random.seed(42)
        self.file_dict = t_files.file_dict
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.train_height = train_height
        self.train_width = train_width
        self.load_multiple_targets_per_file = load_multiple_targets_per_file
        self.max_intenstiy = max_intensity
        self.epoch_done = False
        if self.load_multiple_targets_per_file:
            self.n_multiple_targets = n_multiple_targets
        else:
            self.n_multiple_targets = 1
        self.train_examples = []

        for iglu_movie in self.file_dict.keys():
            path = self.file_dict[iglu_movie]["filepath"]
            movie_len = self.file_dict[iglu_movie]["shape"][0]
            if self.load_multiple_targets_per_file:
                iterate_to = movie_len - sequence_len - 1 - n_multiple_targets
                iterate_step = self.n_multiple_targets
            else:
                iterate_to = movie_len - sequence_len - 1
                iterate_step = 1
            self.train_examples += [
                [path, target]
                for target in range(sequence_len + 1, iterate_to, iterate_step)
            ]
        self.shuffle_array()
        self.X_list = []
        self.y_list = []

    def shuffle_array(self) -> None:
        self.epoch_done = False
        random_order = np.arange(len(self.train_examples))
        np.random.shuffle(random_order)
        self.available_train_examples = [
            self.train_examples[idx] for idx in random_order
        ]

    def crop_pad_img(self, img: np.ndarray) -> np.ndarray:
        _, height, width = img.shape
        diff_height = self.train_height - height
        diff_width = self.train_width - width
        pad = False

        # crop
        if diff_height < 0:
            start = np.random.choice(abs(diff_height))
            stop = height - (abs(diff_height) - start)
            img = img[:, start:stop, :]
        if diff_width < 0:
            start = np.random.choice(abs(diff_width))
            stop = width - (abs(diff_width) - start)
            img = img[:, :, start:stop]

        # pad
        if diff_height > 0:
            pad = True
            up_pad = diff_height // 2
            down_pad = diff_height - diff_height // 2
        else:
            up_pad = 0
            down_pad = 0
        if diff_width > 0:
            pad = True
            left_pad = diff_width // 2
            right_pad = diff_width - diff_width // 2
        else:
            left_pad = 0
            right_pad = 0
        if pad:
            pad_coords = ((0, 0), (up_pad, down_pad), (left_pad, right_pad))
            img = np.pad(
                img,
                pad_coords,  # type: ignore
                mode="reflect",
            )
        return img

    def scale_img(self, img: np.ndarray) -> np.ndarray:
        return np.divide(img, self.max_intenstiy)

    def create_train_example(self, filepath: str, target: int) -> None:
        img = tifffile.imread(filepath)
        img = self.crop_pad_img(img)
        for i in range(self.n_multiple_targets):
            target_ = target + i
            X_tmp = self.scale_img(
                np.append(
                    img[target_ - self.sequence_len - 1 : target_ - 1],
                    img[target_ + 1 : self.sequence_len + 1],
                    axis=0,
                )
            )
            y_tmp = self.scale_img(img[target_].reshape(1, img.shape[1], img.shape[2]))
            self.X_list.append(X_tmp)
            self.y_list.append(y_tmp)

    def get_batch(self) -> bool:
        self.X_list = []
        self.y_list = []
        for _ in range(self.batch_size // self.n_multiple_targets):
            if len(self.available_train_examples) == 0:
                self.epoch_done = True
                return False
            filepath, target = self.available_train_examples.pop(0)
            self.create_train_example(filepath, target)
        self.X = torch.tensor(np.array(self.X_list), dtype=torch.float)
        del self.X_list
        self.y = torch.tensor(np.array(self.y_list), dtype=torch.float)
        del self.y_list
        return True
