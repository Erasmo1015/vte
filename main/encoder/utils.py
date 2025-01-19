import enum
import numpy as np
from numpy.core.numeric import full
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import PIL
from tqdm import trange
import wandb
import os
import pickle
# from grid_world import grid
from typing import Optional, TypeVar
import sys

FONT = ImageFont.load_default()

# FONT = ImageFont.truetype(
#     os.path.join(os.path.dirname(__file__), "asset/fonts/arial.ttf"), 30
# )


def contains_nan(tensor):
    is_nan = torch.isnan(tensor)
    prod = torch.prod(1 - is_nan.float())
    return prod == 0.0


def highlite_boundary(input_data):
    input_data[0, :, 0] = 1.0
    input_data[0, :, 1] = 0.0
    input_data[0, :, 2] = 0.0
    input_data[-1, :, 0] = 1.0
    input_data[-1, :, 1] = 0.0
    input_data[-1, :, 2] = 0.0

    input_data[:, 0, 0] = 1.0
    input_data[:, 0, 1] = 0.0
    input_data[:, 0, 2] = 0.0
    input_data[:, -1, 0] = 1.0
    input_data[:, -1, 1] = 0.0
    input_data[:, -1, 2] = 0.0
    return input_data


def tensor2numpy_img(input_tensor):
    return input_tensor.permute(1, 2, 0).data.cpu().numpy()


def add_number(img, number):
    img = np.uint8(img * 255.0)
    img = PIL.Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((5, 1), str(number), (255, 0, 0), font=FONT)
    return np.asarray(img) / 255.0


def plot_rec(
    init_data_list,
    org_data_list,
    rec_data_list,
    mask_data_list,
    prior_mask_list,
    post_mask_list,
    option_list=None,
):
    # get size
    batch_size, init_size, rgb_size, row_size, col_size = init_data_list.size()
    seq_size = org_data_list.size(1)

    # init pad
    row_pad = np.zeros([1, (col_size + 2) * (seq_size + init_size), 3])
    col_pad = np.zeros([row_size, 1, 3])
    red_block = np.ones([row_size, col_size, 3])
    red_block[:, :, 1:] = 0.0
    blue_block = np.ones([row_size, col_size, 3])
    blue_block[:, :, :2] = 0.0

    # init out image
    output_img = []
    output_mask = []
    for img_idx in range(batch_size):
        org_img_list = []
        rec_img_list = []
        p_mask_list = []
        q_mask_list = []

        # for init image
        for i_idx in range(init_size):
            org_img_list.append(col_pad)
            rec_img_list.append(col_pad)
            org_img_list.append(
                highlite_boundary(tensor2numpy_img(init_data_list[img_idx, i_idx]))
            )
            rec_img_list.append(
                highlite_boundary(tensor2numpy_img(init_data_list[img_idx, i_idx]))
            )
            org_img_list.append(col_pad)
            rec_img_list.append(col_pad)

            p_mask_list.append(col_pad)
            q_mask_list.append(col_pad)
            p_mask_list.append(blue_block)
            q_mask_list.append(red_block)
            p_mask_list.append(col_pad)
            q_mask_list.append(col_pad)

        # for roll out sequence
        for i_idx in range(seq_size):
            # padding
            org_img_list.append(col_pad)
            rec_img_list.append(col_pad)
            p_mask_list.append(col_pad)
            q_mask_list.append(col_pad)

            # data
            if mask_data_list[img_idx, i_idx]:
                org_img_list.append(
                    highlite_boundary(tensor2numpy_img(org_data_list[img_idx, i_idx]))
                )
                rec_img_list.append(
                    highlite_boundary(tensor2numpy_img(rec_data_list[img_idx, i_idx]))
                )
                if option_list is not None:
                    # print('option_list', option_list.shape)
                    # print('org_img_list', len(org_img_list))
                    # print('img_idx, i_idx', img_idx, i_idx)
                    org_img_list[-1] = add_number(
                        org_img_list[-1], option_list[img_idx, i_idx]
                    )
                    rec_img_list[-1] = add_number(
                        rec_img_list[-1], option_list[img_idx, i_idx]
                    )
            else:
                org_img_list.append(tensor2numpy_img(org_data_list[img_idx, i_idx]))
                rec_img_list.append(tensor2numpy_img(rec_data_list[img_idx, i_idx]))

            # mask
            p_mask_list.append(blue_block * prior_mask_list[img_idx, i_idx].item())
            q_mask_list.append(red_block * post_mask_list[img_idx, i_idx].item())

            # padding
            org_img_list.append(col_pad)
            rec_img_list.append(col_pad)
            p_mask_list.append(col_pad)
            q_mask_list.append(col_pad)

        # stack
        org_img_list = np.concatenate(org_img_list, 1)
        rec_img_list = np.concatenate(rec_img_list, 1)
        p_mask_list = np.concatenate(p_mask_list, 1)
        q_mask_list = np.concatenate(q_mask_list, 1)
        output_img.append(
            np.concatenate(
                [row_pad, org_img_list, row_pad, row_pad, rec_img_list, row_pad], 0
            )
        )
        output_mask.append(
            np.concatenate(
                [row_pad, p_mask_list, row_pad, row_pad, q_mask_list, row_pad], 0
            )
        )
    output_img = np.clip(np.concatenate(output_img, 0), 0.0, 1.0)
    output_mask = np.clip(np.concatenate(output_mask, 0), 0.0, 1.0)
    return output_img, output_mask


def plot_gen(init_data_list, gen_data_list, mask_data_list=None, option_list=None):
    # get size
    batch_size, init_size, rgb_size, row_size, col_size = init_data_list.size()
    seq_size = gen_data_list.size(1)

    # init pad
    row_pad = np.zeros([1, (col_size + 2) * (seq_size + init_size), 3])
    col_pad = np.zeros([row_size, 1, 3])

    # init out image
    output_img = []
    for img_idx in range(batch_size):
        gen_img_list = []

        # for init image
        for i_idx in range(init_size):
            gen_img_list.append(col_pad)
            gen_img_list.append(
                highlite_boundary(tensor2numpy_img(init_data_list[img_idx, i_idx]))
            )
            gen_img_list.append(col_pad)

        # for roll out sequence
        for i_idx in range(seq_size):
            # padding
            gen_img_list.append(col_pad)

            # data
            if mask_data_list is not None and mask_data_list[img_idx, i_idx]:
                img = highlite_boundary(tensor2numpy_img(gen_data_list[img_idx, i_idx]))
                if option_list is not None:
                    img = add_number(img, option_list[img_idx, i_idx])
                gen_img_list.append(img)
            else:
                gen_img_list.append(tensor2numpy_img(gen_data_list[img_idx, i_idx]))

            # padding
            gen_img_list.append(col_pad)

        # stack
        gen_img_list = np.concatenate(gen_img_list, 1)
        output_img.append(np.concatenate([row_pad, gen_img_list, row_pad], 0))
    output_img = np.clip(np.concatenate(output_img, 0), 0.0, 1.0)
    return output_img


def _log_marginal_v1(writer, marginal, step, mode="train"):
    for i, p in enumerate(marginal):
        writer.add_scalar("option_{}/option_{}".format(mode, i), p, global_step=step)


def log_train_v1(results, writer, b_idx):
    # compute total loss (mean over steps and seqs)
    train_obs_cost = results["obs_cost"].mean()
    train_kl_abs_cost = results["kl_abs_state"].mean()
    train_kl_obs_cost = results["kl_obs_state"].mean()
    train_kl_mask_cost = results["kl_mask"].mean()

    # log
    writer.add_scalar(
        "train/full_cost",
        train_obs_cost + train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost,
        global_step=b_idx,
    )
    writer.add_scalar("train/obs_cost", train_obs_cost, global_step=b_idx)
    writer.add_scalar(
        "train/kl_full_cost",
        train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost,
        global_step=b_idx,
    )
    writer.add_scalar("train/kl_abs_cost", train_kl_abs_cost, global_step=b_idx)
    writer.add_scalar("train/kl_obs_cost", train_kl_obs_cost, global_step=b_idx)
    writer.add_scalar("train/kl_mask_cost", train_kl_mask_cost, global_step=b_idx)
    writer.add_scalar("train/q_ent", results["p_ent"].mean(), global_step=b_idx)
    writer.add_scalar("train/p_ent", results["q_ent"].mean(), global_step=b_idx)
    writer.add_scalar(
        "train/read_ratio", results["mask_data"].sum(1).mean(), global_step=b_idx
    )
    writer.add_scalar("train/beta", results["beta"], global_step=b_idx)

    log_str = (
        "[%08d] train=elbo:%7.3f, obs_nll:%7.3f, "
        "kl_full:%5.3f, kl_abs:%5.3f, kl_obs:%5.3f, kl_mask:%5.3f, "
        "num_reads:%3.1f, beta: %3.3f, "
        "p_ent: %3.2f, q_ent: %3.2f"
    )
    log_data = [
        b_idx,
        -(train_obs_cost + train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost),
        train_obs_cost,
        train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost,
        train_kl_abs_cost,
        train_kl_obs_cost,
        train_kl_mask_cost,
        results["mask_data"].sum(1).mean(),
        results["beta"],
        results["p_ent"].mean(),
        results["q_ent"].mean(),
    ]
    if "encoding_length" in results:
        coding_length = results["encoding_length"].item()
        log_str += ", coding_len: %3.2f"
        log_data.append(coding_length)
        writer.add_scalar("train/coding_length", coding_length, global_step=b_idx)
    if "marginal" in results:
        _log_marginal_v1(writer, results["marginal"], b_idx, mode="train")
    return log_str, log_data


def log_train(results, writer, b_idx):
    # compute total loss (mean over steps and seqs)
    train_obs_cost = results["obs_cost"].mean()
    train_kl_abs_cost = results["kl_abs_state"].mean()
    train_kl_obs_cost = results["kl_obs_state"].mean()
    train_kl_mask_cost = results["kl_mask"].mean()

    stats = {}
    stats["train/full_cost"] = (
        train_obs_cost + train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost
    )
    stats["train/obs_cost"] = train_obs_cost
    stats["train/kl_full_cost"] = (
        train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost
    )
    stats["train/kl_abs_cost"] = train_kl_abs_cost
    stats["train/kl_obs_cost"] = train_kl_obs_cost
    stats["train/kl_mask_cost"] = train_kl_mask_cost
    stats["train/q_ent"] = results["p_ent"].mean()
    stats["train/p_ent"] = results["q_ent"].mean()
    stats["train/read_ratio"] = results["mask_data"].sum(1).mean()
    stats["train/beta"] = results["beta"]

    log_str = (
        "[%08d] train=elbo:%7.3f, obs_nll:%7.3f, "
        "kl_full:%5.3f, kl_abs:%5.3f, kl_obs:%5.3f, kl_mask:%5.3f, "
        "num_reads:%3.1f, beta: %3.3f, "
        "p_ent: %3.2f, q_ent: %3.2f"
    )
    log_data = [
        b_idx,
        -(train_obs_cost + train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost),
        train_obs_cost,
        train_kl_abs_cost + train_kl_obs_cost + train_kl_mask_cost,
        train_kl_abs_cost,
        train_kl_obs_cost,
        train_kl_mask_cost,
        results["mask_data"].sum(1).mean(),
        results["beta"],
        results["p_ent"].mean(),
        results["q_ent"].mean(),
    ]
    if "encoding_length" in results:
        coding_length = results["encoding_length"].item()
        log_str += ", coding_len: %3.2f"
        log_data.append(coding_length)
        stats["train/coding_length"] = coding_length
    if "coding_len_coeff" in results:
        coding_len_coeff = results["coding_len_coeff"]
        log_str += ", coding_len_coeff: %3.6f"
        log_data.append(coding_len_coeff)
        stats["train/coding_len_coeff"] = coding_len_coeff
    if "marginal" in results:
        for i, p in enumerate(results["marginal"]):
            stats["option_{}/option_{}".format("train", i)] = p
    if "grad_norm" in results:
        grad_norm = results["grad_norm"]
        log_str += ", grad_norm: %3.2f"
        log_data.append(grad_norm)
        stats["train/grad_norm"] = grad_norm
    if "vq_loss_list" in results:
        vq_loss = results["vq_loss_list"]
        log_str += ", vq_loss: %3.2f"
        log_data.append(vq_loss)
        stats["train/vq_loss"] = vq_loss
    if "precision" in results:
        log_str += ", precision: %3.2f"
        log_data.append(results["precision"])
        stats["train/precision"] = results["precision"]
        log_str += ", recall: %3.2f"
        log_data.append(results["recall"])
        stats["train/recall"] = results["recall"]
        log_str += ", f1: %3.2f"
        log_data.append(results["f1"])
        stats["train/f1"] = results["f1"]
    return stats, log_str, log_data


def log_test(results, writer, b_idx):
    # compute total loss (mean over steps and seqs)
    test_obs_cost = results["obs_cost"].mean()
    test_kl_abs_cost = results["kl_abs_state"].mean()
    test_kl_obs_cost = results["kl_obs_state"].mean()
    test_kl_mask_cost = results["kl_mask"].mean()

    stats = {}
    stats["valid/full_cost"] = (
        test_obs_cost + test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost
    )
    stats["valid/obs_cost"] = test_obs_cost
    stats["valid/kl_full_cost"] = (
        test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost
    )
    stats["valid/kl_abs_cost"] = test_kl_abs_cost
    stats["valid/kl_obs_cost"] = test_kl_obs_cost
    stats["valid/kl_mask_cost"] = test_kl_mask_cost
    stats["valid/q_ent"] = results["p_ent"].mean()
    stats["valid/p_ent"] = results["q_ent"].mean()
    stats["valid/read_ratio"] = results["mask_data"].sum(1).mean()
    stats["valid/beta"] = results["beta"]

    log_str = (
        "[%08d] valid=elbo:%7.3f, obs_nll:%7.3f, "
        "kl_full:%5.3f, kl_abs:%5.3f, kl_obs:%5.3f, kl_mask:%5.3f, "
        "num_reads:%3.1f"
    )
    log_data = [
        b_idx,
        -(test_obs_cost + test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost),
        test_obs_cost,
        test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost,
        test_kl_abs_cost,
        test_kl_obs_cost,
        test_kl_mask_cost,
        results["mask_data"].sum(1).mean(),
    ]
    if "encoding_length" in results:
        coding_length = results["encoding_length"].item()
        log_str += ", coding_len: %3.2f"
        log_data.append(coding_length)
        # writer.add_scalar('valid/coding_length', coding_length, global_step=b_idx)
        stats["valid/coding_length"] = coding_length
    if "marginal" in results:
        for i, p in enumerate(results["marginal"]):
            stats["option_{}/option_{}".format("valid", i)] = p
    if "precision" in results:
        log_str += ", precision: %3.2f"
        log_data.append(results["precision"])
        stats["train/precision"] = results["precision"]
        log_str += ", recall: %3.2f"
        log_data.append(results["recall"])
        stats["train/recall"] = results["recall"]
        log_str += ", f1: %3.2f"
        log_data.append(results["f1"])
        stats["train/f1"] = results["f1"]
    return stats, log_str, log_data


def log_test_v1(results, writer, b_idx):
    # compute total loss (mean over steps and seqs)
    test_obs_cost = results["obs_cost"].mean()
    test_kl_abs_cost = results["kl_abs_state"].mean()
    test_kl_obs_cost = results["kl_obs_state"].mean()
    test_kl_mask_cost = results["kl_mask"].mean()

    writer.add_scalar(
        "valid/full_cost",
        test_obs_cost + test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost,
        global_step=b_idx,
    )
    writer.add_scalar("valid/obs_cost", test_obs_cost, global_step=b_idx)
    writer.add_scalar(
        "valid/kl_full_cost",
        test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost,
        global_step=b_idx,
    )
    writer.add_scalar("valid/kl_abs_cost", test_kl_abs_cost, global_step=b_idx)
    writer.add_scalar("valid/kl_obs_cost", test_kl_obs_cost, b_idx)
    writer.add_scalar("valid/kl_mask_cost", test_kl_mask_cost, global_step=b_idx)
    writer.add_scalar(
        "valid/read_ratio", results["mask_data"].sum(1).mean(), global_step=b_idx
    )

    log_str = (
        "[%08d] valid=elbo:%7.3f, obs_nll:%7.3f, "
        "kl_full:%5.3f, kl_abs:%5.3f, kl_obs:%5.3f, kl_mask:%5.3f, "
        "num_reads:%3.1f"
    )
    log_data = [
        b_idx,
        -(test_obs_cost + test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost),
        test_obs_cost,
        test_kl_abs_cost + test_kl_obs_cost + test_kl_mask_cost,
        test_kl_abs_cost,
        test_kl_obs_cost,
        test_kl_mask_cost,
        results["mask_data"].sum(1).mean(),
    ]
    if "encoding_length" in results:
        coding_length = results["encoding_length"].item()
        log_str += ", coding_len: %3.2f"
        log_data.append(coding_length)
        writer.add_scalar("valid/coding_length", coding_length, global_step=b_idx)
    if "marginal" in results:
        _log_marginal_v1(writer, results["marginal"], b_idx, mode="test")
    return log_str, log_data


def preprocess(image, bits=5):
    bins = 2**bits
    image = image * 255.0
    if bits < 8:
        image = torch.floor(image / 2 ** (8 - bits))
    image = image / bins
    image = image + image.new_empty(image.size()).uniform_() / bins
    image = image - 0.5
    return image * 2.0


def postprocess(image, bits=5):
    bins = 2**bits
    image = image / 2.0 + 0.5
    image = torch.floor(bins * image)
    image = image * (255.0 / (bins - 1))
    image = torch.clamp(image, min=0.0, max=255.0) / 255.0
    return image


def concat(*data_list):
    return torch.cat(data_list, 1)


def gumbel_sampling(log_alpha, temp, margin=1e-4):
    noise = log_alpha.new_empty(log_alpha.size()).uniform_(margin, 1 - margin)
    gumbel_sample = -torch.log(-torch.log(noise))
    return torch.div(log_alpha + gumbel_sample, temp)


def log_density_concrete(log_alpha, log_sample, temp):
    exp_term = log_alpha - temp * log_sample
    log_prob = torch.sum(exp_term, -1) - 2.0 * torch.logsumexp(exp_term, -1)
    return log_prob


class ToyDataset(Dataset):
    def __init__(self, length, partition, path="./data/toy_data.npy"):
        self.partition = partition
        dataset = np.load(path)
        num_seqs = int(dataset.shape[0] * 0.8)
        if self.partition == "train":
            self.state = dataset[:num_seqs].transpose(0, 3, 2, 1) / 255.0
        else:
            self.state = dataset[num_seqs:].transpose(0, 3, 2, 1) / 255.0
        # processing for making sure divisible by 3
        new_len = len(self.state) // 96 * 96
        self.state = self.state[:new_len]

        self.state = self.state.reshape(-1, 96, 3, 32, 32)
        self.length = length
        self.full_length = self.state.shape[1]
        assert self.full_length % 3 == 0
        assert self.length % 3 == 0

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, index):
        idx0 = np.random.randint(0, (self.full_length - self.length) // 3) * 3
        idx1 = idx0 + self.length

        state = self.state[index, idx0:idx1].astype(np.float32)
        return state


def full_dataloader_toy(
    seq_size, init_size, batch_size, test_size=16, path="./data/toy_data.npy"
):
    train_loader = ToyDataset(
        length=seq_size + init_size * 2, partition="train", path=path
    )
    test_loader = ToyDataset(
        length=seq_size + init_size * 2, partition="test", path=path
    )
    train_loader = DataLoader(dataset=train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_loader, batch_size=test_size, shuffle=False)
    return train_loader, test_loader


# class GridworldDataset(Dataset):
#     def __init__(self, partition, path):
#         self.partition = partition
#         trajectories = []
#         for f_name in sorted(os.listdir(path)):
#             full_path = os.path.join(path, f_name)
#             with open(full_path, "rb") as f:
#                 trajectory = pickle.load(f)
#                 padding = tuple(trajectory[0][i] * 0 for i in range(len(trajectory[0])))
#                 trajectory.append(padding)
#                 trajectory = [padding] + trajectory
#                 trajectories.append(trajectory)
#         num_heldout = 1
#         if self.partition == "train":
#             self.state = trajectories[
#                 :-num_heldout
#             ]  # num_train x ep length x (s, a, s_tp1)
#         else:
#             self.state = trajectories[-num_heldout:]

#         self.obs_size = self.state[0][0][0].shape
#         self.action_size = 9

#     @property
#     def seq_size(self):
#         return len(self.state[0]) - 2

#     def __len__(self):
#         return len(self.state)

#     def __getitem__(self, index):
#         traj = self.state[index]
#         s, a, _ = zip(*traj)
#         return np.stack(s), np.stack(a)


# def gridworld_loader(batch_size, path="./data/demos"):
#     train_dataset = GridworldDataset(partition="train", path=path)
#     test_dataset = GridworldDataset(partition="test", path=path)
#     train_loader = DataLoader(
#         dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
#     )
#     test_loader = DataLoader(
#         dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
#     )
#     return train_loader, test_loader


class ComPILEDataset(Dataset):
    def __init__(self, partition, expert_file=""):
        sys.path.append('grid_world')
        # trajectories = np.load("compile.npy", allow_pickle=True)
        trajectories = np.load(expert_file, allow_pickle=True)
        self.partition = partition
        num_heldout = 100
        if self.partition == "train":
            self.state = trajectories[
                :-num_heldout
            ]  # num_train x ep length x (s, a, s_tp1)
        else:
            self.state = trajectories[-num_heldout:]
        level_placeholder = 0
        level = [level_placeholder for i in range(len(self.state))]
        self.level = np.array(level, dtype='int')
        self.obs_size = self.state[0][0][0].shape
        # self.action_size = len(grid.Action)

    @property
    def seq_size(self):
        return len(self.state[0]) - 2

    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        traj = self.state[index]
        s, a, _ = zip(*traj)
        return np.stack(s).astype(np.float32), np.stack(a), self.level[index]


def compile_loader(batch_size, expert_file=""):
    train_dataset = ComPILEDataset(partition="train", expert_file=expert_file)
    test_dataset = ComPILEDataset(partition="test", expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader


class MiniWorldDataset(Dataset):
    def __init__(self, partition, stack_n=0):
        with open("world3d.pkl", "rb") as f:
            trajectories = pickle.load(f)
        reformatted_trajectories = []
        for states, actions in zip(trajectories["states"], trajectories["actions"]):
            states = list(zip(*states))[0]
            traj = list(zip(states[:-1], actions, states[1:]))
            pad = (states[0] * 0, actions[0] * 0, states[0] * 0)
            trajectory = [pad] + traj + [pad]
            if stack_n > 0:
                trajectory = self._stack(trajectory, stack_n)
            reformatted_trajectories.append(trajectory)
        trajectories = reformatted_trajectories
        self.partition = partition

        num_heldout = 100
        if self.partition == "train":
            self.state = trajectories[
                :-num_heldout
            ]  # num_train x ep length x (s, a, s_tp1)
        else:
            self.state = trajectories[-num_heldout:]

        self.obs_size = self.state[0][0][0].shape
        self.action_size = 5

    @property
    def seq_size(self):
        return len(self.state[0]) - 2

    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        traj = self.state[index]
        s, a, _ = zip(*traj)
        return np.stack(s).astype(np.float32), np.stack(a)

    def _stack(self, trajectories, stack_n):
        pad_s, _, pad_s_tp1 = trajectories[0]
        last_s_t = [pad_s] * stack_n
        last_s_tp1 = [pad_s_tp1] * stack_n
        new_trajectory = []
        for t in trajectories:
            s_t, a_t, s_tp1 = t
            last_s_t = last_s_t[1:] + [s_t]
            last_s_tp1 = last_s_tp1[1:] + [s_tp1]
            stacked_s_t = np.concatenate(last_s_t, axis=-1)
            stacked_s_tp1 = np.concatenate(last_s_tp1, axis=-1)
            new_trajectory.append((stacked_s_t, a_t, stacked_s_tp1))
        return new_trajectory


def miniworld_loader(batch_size):
    train_dataset = MiniWorldDataset(partition="train", stack_n=0)
    test_dataset = MiniWorldDataset(partition="test", stack_n=0)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader


class HilDataset(Dataset):
    def __init__(self, partition, seq_size=1000):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/adapt-image-models/love/data_test")
        dataset_paths = []
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0][0].shape[0]
        self.action_size = self.action[0][0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]
    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def hil_loader(batch_size, seq_size):
    train_dataset = HilDataset(partition="train", seq_size=seq_size)
    test_dataset = HilDataset(partition="test", seq_size=seq_size)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def hil_full_loader(batch_size):
    full_dataset = HilDataset(partition="full")
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class CheetahDataset(Dataset):
    def __init__(self, partition, seq_size=1000,
                 expert_file='/home/zichang/proj/IQ-Learn/iq_learn/experts/HalfCheetah-v2_25.pkl'):
        mycwd = os.getcwd()
        if "encoder" not in mycwd:
            newcwd = mycwd + "/encoder/"
            newcwd = '/home/zichang/proj/IQ-Learn/iq_learn/encoder' # for server2
            # newcwd = '/common/home/users/z/zichang.ge.2023/proj/IQ-Learn/iq_learn/encoder' # for origami server
            os.chdir(newcwd)
            print(f"Dataset working in: {newcwd}")
        # os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
        dataset_paths = expert_file.split(",")
        # dataset_paths = [expert_file]
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]
    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def cheetah_loader(batch_size, seq_size):
    train_dataset = CheetahDataset(partition="train", seq_size=seq_size)
    test_dataset = CheetahDataset(partition="test", seq_size=seq_size)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def cheetah_full_loader(batch_size, expert_file, seq_dim=1000):
    full_dataset = CheetahDataset(partition="full", seq_size=seq_dim, expert_file=expert_file, )
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class CartpoleDataset(Dataset):
    def __init__(self, partition, seq_size=500,
                expert_file="/home/zichang/proj/IQ-Learn/iq_learn/experts/Cartpole-v0_25_500r.pkl"):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
        # split expert_file by "," and make it an array to dataset_paths
        dataset_paths = expert_file.split(",")
        # dataset_paths = [expert_file]
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = 1
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 50 4
        a = self.action[index][:max_len]
        # a 50
        a = np.expand_dims(a, axis=-1)
        # a 50 1
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]
    
    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        # print mean, max, min of lengths
        # print("mean max min:",np.mean(trajectories['lengths']), np.max(trajectories['lengths']),np.min(trajectories['lengths']))
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def cartpole_loader(batch_size, seq_size):
    train_dataset = CartpoleDataset(partition="train", seq_size=seq_size)
    test_dataset = CartpoleDataset(partition="test", seq_size=seq_size)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def cartpole_full_loader(batch_size, expert_file):
    full_dataset = CartpoleDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class LunarDataset(Dataset):
    def __init__(self, partition, seq_size=500,
                expert_file="/home/zichang/proj/IQ-Learn/iq_learn/experts/LunarLander-v2_100_84r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/experts/LunarLander-v2_100_180r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/experts/LunarLander-v2_100_230r.pkl"):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
        # split expert_file by "," and make it an array to dataset_paths
        dataset_paths = expert_file.split(",")
        # dataset_paths = [expert_file]
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = 1
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 50 4
        a = self.action[index][:max_len]
        # a 50
        a = np.expand_dims(a, axis=-1)
        # a 50 1
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]
    
    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        # print mean, max, min of lengths
        # print("mean max min:",np.mean(trajectories['lengths']), np.max(trajectories['lengths']),np.min(trajectories['lengths']))
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def lunar_loader(batch_size, seq_size):
    train_dataset = LunarDataset(partition="train", seq_size=seq_size)
    test_dataset = LunarDataset(partition="test", seq_size=seq_size)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def lunar_full_loader(batch_size, expert_file):
    full_dataset = LunarDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class HopperDataset(Dataset):
    def __init__(self, partition, seq_size=1000,
                 expert_file=""):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder/")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]
    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def hopper_loader(batch_size, seq_size, 
                  expert_file="/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/hopper/Hopper-v2_100_299r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/hopper/Hopper-v2_100_803r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/hopper/Hopper-v2_100_1404r.pkl"):
    train_dataset = HopperDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = HopperDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def hopper_full_loader(batch_size, expert_file):
    full_dataset = HopperDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class WalkerDataset(Dataset):
    def __init__(self, partition, seq_size=1000,
                 expert_file=""):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder/")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]
    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def walker_loader(batch_size, seq_size, 
                  expert_file=""):
    train_dataset = WalkerDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = WalkerDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def walker_full_loader(batch_size, expert_file):
    full_dataset = WalkerDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class AntDataset(Dataset):
    def __init__(self, partition, seq_size=1000,
                 expert_file=""):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder/")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]
    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def ant_loader(batch_size, seq_size, 
                  expert_file=""):
    train_dataset = AntDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = AntDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def ant_full_loader(batch_size, expert_file):
    full_dataset = AntDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class HumanoidDataset(Dataset):
    def __init__(self, partition, seq_size=1000,
                 expert_file=""):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder/")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]
    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def humanoid_loader(batch_size, seq_size, 
                  expert_file=""):
    train_dataset = HumanoidDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = HumanoidDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def humanoid_full_loader(batch_size, expert_file):
    full_dataset = HumanoidDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class PusherDataset(Dataset):
    def __init__(self, partition, seq_size=1000,
                 expert_file=""):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder/")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]

    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def pusher_loader(batch_size, seq_size, 
                  expert_file="/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/pusher/Swimmer-v4_100_299r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/swimmer/Swimmer-v4_100_803r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/swimmer/Swimmer-v4_100_1404r.pkl"):
    train_dataset = PusherDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = PusherDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def pusher_full_loader(batch_size, expert_file):
    full_dataset = PusherDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class SwimmerDataset(Dataset):
    def __init__(self, partition, seq_size=1000,
                 expert_file=""):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder/")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]

    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def swimmer_loader(batch_size, seq_size, 
                  expert_file="/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/swimmer/Swimmer-v4_100_299r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/swimmer/Swimmer-v4_100_803r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/swimmer/Swimmer-v4_100_1404r.pkl"):
    train_dataset = SwimmerDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = SwimmerDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def swimmer_full_loader(batch_size, expert_file):
    full_dataset = SwimmerDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class HumanoidStandupDataset(Dataset):
    def __init__(self, partition, seq_size=1000,
                 expert_file=""):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder/")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]

    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def humanoid_standup_loader(batch_size, seq_size, 
                  expert_file="/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/swimmer/Swimmer-v4_100_299r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/swimmer/Swimmer-v4_100_803r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/swimmer/Swimmer-v4_100_1404r.pkl"):
    train_dataset = HumanoidStandupDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = HumanoidStandupDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def humanoid_standup_full_loader(batch_size, expert_file):
    full_dataset = HumanoidStandupDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class InvertedpDataset(Dataset):
    def __init__(self, partition, seq_size=1000,
                 expert_file=""):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder/")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size


    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        # s 1000 17
        a = np.squeeze(self.action[index])[:max_len]
        # a 1000 6
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]

    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action
    
def invertedp_loader(batch_size, seq_size, 
                     expert_file="/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/invertedp/InvertedPendulum-v2_100_299r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/invertedp/InvertedPendulum-v2_100_803r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/invertedp/InvertedPendulum-v2_100_1404r.pkl"):
    train_dataset = InvertedpDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = InvertedpDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def invertedp_full_loader(batch_size, expert_file):
    full_dataset = InvertedpDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class LunarLanderDataset(Dataset):
    def __init__(self, partition, seq_size=1000, expert_file=""):
        mycwd = os.getcwd()
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder/")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size = 1
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size

    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        a = np.squeeze(self.action[index])[:max_len]
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]

    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action

def lunarlander_loader(batch_size, seq_size, 
                       expert_file="/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/lunarlander/LunarLander-v2_100_299r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/lunarlander/LunarLander-v2_100_803r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/lunarlander/LunarLander-v2_100_1404r.pkl"):
    train_dataset = LunarLanderDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = LunarLanderDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def lunarlander_full_loader(batch_size, expert_file):
    full_dataset = LunarLanderDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

def preprocess_d4rl_episodes_from_path(path: str, max_episode_length: int, number: Optional[int] = None, proportion: Optional[float] = None):
    '''
    read dataset from pickle file
    '''
    with open(path, 'rb') as f:
        episodes = pickle.load(f)

    n_episode = len(episodes)
    
    print(f'Loading dataset from {path}: {n_episode} episodes')

    episode_lengths = [e['rewards'].shape[0] for e in episodes]
    key_dims = {key: episodes[0][key].shape[-1] for key in episodes[0].keys()}
    buffers = {key: np.zeros((n_episode, max_episode_length, key_dims[key]), dtype=np.float32) for key in episodes[0].keys()}

    for idx, e in enumerate(episodes):  # put episodes into fix sized numpy arrays (for padding)
        for key, buffer in buffers.items():
            buffer[idx, :episode_lengths[idx]] = e[key]
    
    buffers['episode_lengths'] = np.array(episode_lengths)

    return buffers

def preprocess_gcpc_to_love_format(path: str, max_episode_length: int):
    '''
    Read GCPC dataset and return states and actions in LOVE-compatible format.
    '''
    with open(path, 'rb') as f:
        episodes = pickle.load(f)

    n_episode = len(episodes)
    
    print(f'Loading dataset from {path}: {n_episode} episodes')

    # Extract the key dimensions for padding (optional)
    episode_lengths = [e['rewards'].shape[0] for e in episodes]
    
    # Initialize lists to store states and actions
    states = []
    actions = []

    # Loop through the episodes and collect states and actions
    for e in episodes:
        states.append(np.array(e['observations'], dtype=np.float32))
        actions.append(np.array(e['actions'], dtype=np.float32))

    # Return states and actions in the format expected by LOVE
    return states, actions

class AntMazeDataset(Dataset):
    def __init__(self, partition, seq_size=1000, expert_file=""):
        mycwd = os.getcwd()
        print(f"Dataset working in: {mycwd}")
        if "encoder" not in mycwd:
            newcwd = mycwd + "/encoder/"
            os.chdir(newcwd)
            print(f"Dataset working in: {newcwd}")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            # state_i, action_i = preprocess_gcpc_to_love_format(dataset_path, 5000)
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size =  self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size

    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        a = np.squeeze(self.action[index])[:max_len]
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]

    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        print(f"state shape: {trajectories['states'][0].shape}")
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action

def antmaze_loader(batch_size, seq_size, 
                       expert_file="/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/lunarlander/LunarLander-v2_100_299r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/lunarlander/LunarLander-v2_100_803r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/lunarlander/LunarLander-v2_100_1404r.pkl"):
    train_dataset = AntMazeDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = AntMazeDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def antmaze_full_loader(batch_size, expert_file):
    full_dataset = AntMazeDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader

class KitchenDataset(Dataset):
    def __init__(self, partition, seq_size=1000, expert_file=""):
        mycwd = os.getcwd()
        print(f"Dataset working in: {mycwd}")
        if "encoder" not in mycwd:
            newcwd = mycwd + "/encoder/"
            os.chdir(newcwd)
            print(f"Dataset working in: {newcwd}")
        dataset_paths = expert_file.split(",")
        state = []
        action = []
        level = []
        for idx, dataset_path in enumerate(dataset_paths):
            # state_i, action_i = preprocess_gcpc_to_love_format(dataset_path, 5000)
            state_i, action_i = self.open_dataset(dataset_path)
            level_i = [idx for i in range(len(state_i))]
            state.extend(state_i)
            action.extend(action_i)
            level.extend(level_i)
        
        os.chdir(mycwd) 
        state = np.array(state, dtype='object')
        action = np.array(action, dtype='object')
        level = np.array(level, dtype='int')
        self.partition = partition
        num_heldout = 10
        if self.partition == "train":
            self.state = state[:-num_heldout] 
            self.action = action[:-num_heldout]  
            self.level = level[:-num_heldout]  
        elif self.partition == "test":
            self.state = state[-num_heldout:]
            self.action = action[-num_heldout:]
            self.level = level[:-num_heldout]  
        else:
            self.state = state[:]
            self.action = action[:]
            self.level = level[:]  
        self.obs_size = self.state[0][0].shape[0]
        self.action_size =  self.action[0][0].shape[0]
        self._num_levels = len(dataset_paths)
        self._seq_size = seq_size

    @property
    def seq_size(self):
        return self._seq_size - 2

    @property
    def num_levels(self):
        return self._num_levels
    
    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        max_len = self._seq_size
        s = np.squeeze(self.state[index])[:max_len]
        a = np.squeeze(self.action[index])[:max_len]
        return np.stack(s).astype(np.float32), np.stack(a).astype(np.float32), self.level[index]

    def open_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        print(f"state shape: {trajectories['states'][0].shape}")
        state = [np.array(x, dtype=type) for x in trajectories['states']]
        action = [np.array(x, dtype=type) for x in trajectories['actions']]
        return state, action

def kitchen_loader(batch_size, seq_size, 
                       expert_file="/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/lunarlander/LunarLander-v2_100_299r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/lunarlander/LunarLander-v2_100_803r.pkl,/home/zichang/proj/IQ-Learn/iq_learn/encoder/expert/lunarlander/LunarLander-v2_100_1404r.pkl"):
    train_dataset = KitchenDataset(partition="train", seq_size=seq_size, expert_file=expert_file)
    test_dataset = KitchenDataset(partition="test", seq_size=seq_size, expert_file=expert_file)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader

def kitchen_full_loader(batch_size, expert_file):
    full_dataset = KitchenDataset(partition="full", expert_file=expert_file)
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=batch_size, shuffle=False
    )
    return full_loader