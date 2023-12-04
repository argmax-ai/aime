import os
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler


class ArrayDict(dict):
    def vmap_(self, fn, rewrite=True):
        for k, v in self.items():
            result = fn(v)
            if rewrite:
                self[k] = result
        return self

    def expand_dim_equal_(
        self, black_list=["image", "frontview_image", "agentview_image"]
    ):
        # TODO: logic is wrong if there is image data in the dict
        max_length = max([len(v.shape) for k, v in self.items() if k not in black_list])
        for k, v in self.items():
            if k in black_list:
                continue
            if len(v.shape) < max_length:
                for _ in range(max_length - len(v.shape)):
                    v = v[..., None]
                self[k] = v
        return self

    def __len__(self) -> int:
        lengths = [len(v) for v in self.values()]
        assert np.all([n == lengths[0] for n in lengths])
        return lengths[0]

    def __getitem__(self, index):
        if isinstance(index, str):
            return dict.__getitem__(self, index)
        else:
            return ArrayDict({k: v[index] for k, v in self.items()})

    def to(self, target: Union[str, torch.Tensor]):
        return self.vmap_(lambda v: v.to(target))

    def to_torch(self):
        return self.vmap_(lambda v: torch.tensor(v))

    def to_numpy(self):
        return self.vmap_(lambda v: v.detach().cpu().numpy())

    def to_float_torch(self):
        return self.vmap_(lambda v: v.float())

    def get_type(self):
        return type(list(self.values())[0])

    @classmethod
    def merge_list(cls, array_dicts: List["ArrayDict"], merge_fn) -> "ArrayDict":
        keys = array_dicts[0].keys()
        return ArrayDict(
            {k: merge_fn([array_dict[k] for array_dict in array_dicts]) for k in keys}
        )

    @classmethod
    def stack(cls, array_dicts: List["ArrayDict"], dim: int) -> "ArrayDict":
        if array_dicts[0].get_type() is torch.Tensor:
            merge_fn = partial(torch.stack, dim=dim)
        else:
            merge_fn = partial(np.stack, axis=dim)
        return cls.merge_list(array_dicts, merge_fn)

    @classmethod
    def cat(cls, array_dicts: List["ArrayDict"], dim: int) -> "ArrayDict":
        if array_dicts[0].get_type() is torch.Tensor:
            merge_fn = partial(torch.cat, dim=dim)
        else:
            merge_fn = partial(np.concatenate, axis=dim)
        return cls.merge_list(array_dicts, merge_fn)


class SequenceDataset(Dataset):
    def __init__(
        self, root: str, horizon: int, overlap: bool, max_capacity: Optional[int] = None
    ) -> None:
        super().__init__()
        self.root = root
        self.horizon = horizon
        self.overlap = overlap
        self.max_capacity = max_capacity
        self.loaded_file = []
        self.trajectories = []
        self.index_lookup = []

        self.update()  # call update to do the initialization

    def update(self):
        self._update_trajectories()
        self._update_index_map()

    def _update_trajectories(self):
        raise NotImplementedError

    def _update_index_map(self):
        trajectory_index = (
            0
            if len(self.index_lookup) == 0
            else max([pair[0] for pair in self.index_lookup]) + 1
        )
        while trajectory_index < len(self.trajectories):
            trajectory = self.trajectories[trajectory_index]

            total_clip = (
                len(trajectory) // self.horizon
                if not self.overlap
                else max(len(trajectory) - self.horizon + 1, 0)
            )

            for i in range(total_clip):
                time_index = i * self.horizon if not self.overlap else i
                self.index_lookup.append((trajectory_index, time_index))

            trajectory_index += 1

    def keep(self, num_trajectories: int, random: bool = False):
        """keep a subset of the dataset, in the forward order"""
        if num_trajectories >= len(self.trajectories):
            return
        selected_index = np.arange(len(self.trajectories))
        if random:
            np.random.shuffle(selected_index)
        selected_index = selected_index[:num_trajectories]
        selected_index = list(selected_index)
        self.trajectories = [self.trajectories[index] for index in selected_index]
        self.loaded_file = [self.loaded_file[index] for index in selected_index]
        self.index_lookup = [
            (trajectory_index, time_index)
            for trajectory_index, time_index in self.index_lookup
            if trajectory_index in selected_index
        ]

    def __len__(self):
        return len(self.index_lookup)

    def __getitem__(self, index):
        trajectory_index, time_index = self.index_lookup[index]
        return self.trajectories[trajectory_index][
            time_index : time_index + self.horizon
        ]
    
    @classmethod
    def sort(cls, file_list):
        return sorted(file_list, key=lambda file_name: int(file_name.split('.')[0]))


class NPZFolder(SequenceDataset):
    def _update_trajectories(self):
        file_list = self.sort(os.listdir(self.root))
        if self.max_capacity is not None:
            file_list = file_list[: self.max_capacity]
        for file in file_list:
            if file in self.loaded_file:
                continue
            self.loaded_file.append(file)
            file = os.path.join(self.root, file)
            data = ArrayDict(np.load(file))
            data.expand_dim_equal_()
            data.to_torch()
            data.to_float_torch()
            for k, v in data.items():
                if len(v.shape) == 4:
                    data[k] = v.permute(0, 3, 1, 2).contiguous() / 255
            self.trajectories.append(data)


def get_epoch_loader(
    dataset: SequenceDataset, batch_size: int, shuffle: bool, num_workers: int = 2
):
    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        collate_fn=partial(ArrayDict.stack, dim=1),
        num_workers=num_workers,
    )


def get_sample_loader(
    dataset: SequenceDataset, batch_size: int, batchs: int, num_workers: int = 2
):
    return DataLoader(
        dataset,
        batch_size,
        collate_fn=partial(ArrayDict.stack, dim=1),
        num_workers=num_workers,
        sampler=RandomSampler(
            dataset, replacement=True, num_samples=batchs * batch_size
        ),
    )
