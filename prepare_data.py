# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os

import numpy as np
import torch

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from tqdm import tqdm
from utils import utils_transform


def main(args, bm):
    for dataroot_subset in ["BioMotionLab_NTroje", "CMU", "MPI_HDM05"]:
        print(dataroot_subset)
        for phase in ["train", "test"]:
            print(phase)
            savedir = os.path.join(args.save_dir, dataroot_subset, phase)
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            split_file = os.path.join(
                "prepare_data/data_split", dataroot_subset, phase + "_split.txt"
            )

            with open(split_file, "r") as f:
                filepaths = [line.strip() for line in f]

            rotation_local_full_gt_list = []
            hmd_position_global_full_gt_list = []
            body_parms_list = []
            head_global_trans_list = []

            idx = 0
            for filepath in tqdm(filepaths):
                data = {}
                bdata = np.load(
                    os.path.join(args.root_dir, filepath), allow_pickle=True
                )

                if "mocap_framerate" in bdata:
                    framerate = bdata["mocap_framerate"]
                else:
                    continue
                idx += 1

                if framerate == 120:
                    stride = 2
                elif framerate == 60:
                    stride = 1
                else:
                    raise AssertionError(
                        "Please check your AMASS data, should only have 2 types of framerate, either 120 or 60!!!"
                    )

                bdata_poses = bdata["poses"][::stride, ...]
                bdata_trans = bdata["trans"][::stride, ...]
                subject_gender = bdata["gender"]

                body_parms = {
                    "root_orient": torch.Tensor(
                        bdata_poses[:, :3]
                    ),  # .to(comp_device), # controls the global root orientation
                    "pose_body": torch.Tensor(
                        bdata_poses[:, 3:66]
                    ),  # .to(comp_device), # controls the body
                    "trans": torch.Tensor(
                        bdata_trans
                    ),  # .to(comp_device), # controls the global body position
                }

                body_parms_list = body_parms

                body_pose_world = bm(
                    **{
                        k: v
                        for k, v in body_parms.items()
                        if k in ["pose_body", "root_orient", "trans"]
                    }
                )

                output_aa = torch.Tensor(bdata_poses[:, :66]).reshape(-1, 3)
                output_6d = utils_transform.aa2sixd(output_aa).reshape(
                    bdata_poses.shape[0], -1
                )
                rotation_local_full_gt_list = output_6d[1:]

                rotation_local_matrot = aa2matrot(
                    torch.tensor(bdata_poses).reshape(-1, 3)
                ).reshape(bdata_poses.shape[0], -1, 9)
                rotation_global_matrot = local2global_pose(
                    rotation_local_matrot, bm.kintree_table[0].long()
                )  # rotation of joints relative to the origin

                head_rotation_global_matrot = rotation_global_matrot[:, [15], :, :]

                rotation_global_6d = utils_transform.matrot2sixd(
                    rotation_global_matrot.reshape(-1, 3, 3)
                ).reshape(rotation_global_matrot.shape[0], -1, 6)
                input_rotation_global_6d = rotation_global_6d[1:, [15, 20, 21], :]

                rotation_velocity_global_matrot = torch.matmul(
                    torch.inverse(rotation_global_matrot[:-1]),
                    rotation_global_matrot[1:],
                )
                rotation_velocity_global_6d = utils_transform.matrot2sixd(
                    rotation_velocity_global_matrot.reshape(-1, 3, 3)
                ).reshape(rotation_velocity_global_matrot.shape[0], -1, 6)
                input_rotation_velocity_global_6d = rotation_velocity_global_6d[
                    :, [15, 20, 21], :
                ]

                position_global_full_gt_world = body_pose_world.Jtr[
                    :, :22, :
                ]  # position of joints relative to the world origin

                position_head_world = position_global_full_gt_world[
                    :, 15, :
                ]  # world position of head

                head_global_trans = torch.eye(4).repeat(
                    position_head_world.shape[0], 1, 1
                )
                head_global_trans[:, :3, :3] = head_rotation_global_matrot.squeeze()
                head_global_trans[:, :3, 3] = position_global_full_gt_world[:, 15, :]

                head_global_trans_list = head_global_trans[1:]

                num_frames = position_global_full_gt_world.shape[0] - 1

                hmd_position_global_full_gt_list = torch.cat(
                    [
                        input_rotation_global_6d.reshape(num_frames, -1),
                        input_rotation_velocity_global_6d.reshape(num_frames, -1),
                        position_global_full_gt_world[1:, [15, 20, 21], :].reshape(
                            num_frames, -1
                        ),
                        position_global_full_gt_world[1:, [15, 20, 21], :].reshape(
                            num_frames, -1
                        )
                        - position_global_full_gt_world[:-1, [15, 20, 21], :].reshape(
                            num_frames, -1
                        ),
                    ],
                    dim=-1,
                )

                data["rotation_local_full_gt_list"] = rotation_local_full_gt_list
                data[
                    "hmd_position_global_full_gt_list"
                ] = hmd_position_global_full_gt_list
                data["body_parms_list"] = body_parms_list
                data["head_global_trans_list"] = head_global_trans_list
                data["position_global_full_gt_world"] = (
                    position_global_full_gt_world[1:].cpu().float()
                )
                data["framerate"] = 60
                data["gender"] = subject_gender
                data["filepath"] = filepath

                torch.save(data, os.path.join(savedir, "{}.pt".format(idx)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default=None,
        help="=dir where you put your smplh and dmpls dirs",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", type=str, default=None, help="=dir where you put your AMASS data"
    )
    args = parser.parse_args()

    # Here we follow the AvatarPoser paper and use male model for all sequences
    bm_fname_male = os.path.join(args.support_dir, "smplh/{}/model.npz".format("male"))
    dmpl_fname_male = os.path.join(
        args.support_dir, "dmpls/{}/model.npz".format("male")
    )

    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    bm_male = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname_male,
    )

    main(args, bm_male)
