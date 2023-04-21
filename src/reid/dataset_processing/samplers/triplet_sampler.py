#
# Copied on Tue Nov 15 2022 by Nguyen Qui Vinh Quang
#
# @licensed: Computer Vision & Image Processing Lab
# @project: VehicleReid
#

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        count = 0
        self.final_idxs = self._get_idxs()
        self.length = len(self.final_idxs)

    def _get_idxs(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return final_idxs

    def __iter__(self):
        return iter(self.final_idxs)

    def __len__(self):
        return self.length


# New add by gu
class RandomIdentitySampler_alignedreid(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


class BatchSampler(Sampler):
    """
    sampler used in dataloader. method __iter__ should output the indices each time it is called
    """

    def __init__(self, dataset, n_class, n_num, *args, **kwargs):
        super(BatchSampler, self).__init__(dataset, *args, **kwargs)
        self.n_class = n_class
        self.n_num = n_num
        self.batch_size = n_class * n_num
        self.dataset = dataset
        self.labels = np.array(dataset.lb_ids)
        self.labels_uniq = np.array(list(dataset.lb_ids_uniq))
        self.len = len(dataset) // self.batch_size
        self.lb_img_dict = dataset.lb_img_dict
        self.iter_num = len(self.labels_uniq) // self.n_class

    def __iter__(self):
        curr_p = 0
        np.random.shuffle(self.labels_uniq)
        for k, v in self.lb_img_dict.items():
            random.shuffle(self.lb_img_dict[k])
        for i in range(self.iter_num):
            label_batch = self.labels_uniq[curr_p : curr_p + self.n_class]
            curr_p += self.n_class
            idx = []
            for lb in label_batch:
                if len(self.lb_img_dict[lb]) > self.n_num:
                    idx_smp = np.random.choice(
                        self.lb_img_dict[lb], self.n_num, replace=False
                    )
                else:
                    idx_smp = np.random.choice(
                        self.lb_img_dict[lb], self.n_num, replace=True
                    )
                idx.extend(idx_smp.tolist())
            yield idx

    def __len__(self):
        return self.iter_num
