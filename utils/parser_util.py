# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import json
import os
from argparse import ArgumentParser


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)
    for a in args_to_overwrite:
        if a in model_args.keys():
            # Use the chosen dataset, or use the dataset that is used to train the model
            if a == "dataset":
                if args.__dict__[a] is None:
                    args.__dict__[a] = model_args[a]
            elif a == "input_motion_length":
                continue
            else:
                args.__dict__[a] = model_args[a]
        else:
            print(
                "Warning: was not able to load [{}], using default value [{}] instead.".format(
                    a, args.__dict__[a]
                )
            )
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError("group_name was not found.")


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except Exception:
        raise ValueError("model_path argument must be specified.")


def add_base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument(
        "--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU."
    )
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument(
        "--batch_size", default=64, type=int, help="Batch size during training."
    )
    group.add_argument(
        "--timestep_respacing", default="", type=str, help="ddim timestep respacing."
    )


def add_diffusion_options(parser):
    group = parser.add_argument_group("diffusion")
    group.add_argument(
        "--noise_schedule",
        default="cosine",
        choices=["linear", "cosine"],
        type=str,
        help="Noise schedule type",
    )
    group.add_argument(
        "--diffusion_steps",
        default=1000,
        type=int,
        help="Number of diffusion steps (denoted T in the paper)",
    )
    group.add_argument(
        "--sigma_small", default=True, type=bool, help="Use smaller sigma values."
    )


def add_model_options(parser):
    group = parser.add_argument_group("model")
    group.add_argument(
        "--arch",
        default="DiffMLP",
        type=str,
        help="Architecture types as reported in the paper.",
    )
    group.add_argument(
        "--motion_nfeat", default=132, type=int, help="motion feature dimension"
    )
    group.add_argument(
        "--sparse_dim", default=54, type=int, help="sparse signal feature dimension"
    )
    group.add_argument("--layers", default=8, type=int, help="Number of layers.")
    group.add_argument(
        "--latent_dim", default=512, type=int, help="Transformer/GRU width."
    )
    group.add_argument(
        "--cond_mask_prob",
        default=0.0,
        type=float,
        help="The probability of masking the condition during training."
        " For classifier-free guidance learning.",
    )
    group.add_argument(
        "--input_motion_length",
        default=196,
        type=int,
        help="Limit for the maximal number of frames.",
    )
    group.add_argument(
        "--no_normalization",
        action="store_true",
        help="no data normalisation for the 6d motions",
    )


def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--dataset",
        default=None,
        choices=[
            "amass",
        ],
        type=str,
        help="Dataset name (choose from list).",
    )
    group.add_argument(
        "--dataset_path",
        default="./dataset/AMASS/",
        type=str,
        help="Dataset path",
    )


def add_training_options(parser):
    group = parser.add_argument_group("training")
    group.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="Path to save checkpoints and results.",
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="If True, will enable to use an already existing save_dir.",
    )
    group.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"],
        type=str,
        help="Choose platform to log results. NoPlatform means no logging.",
    )
    group.add_argument("--lr", default=2e-4, type=float, help="Learning rate.")
    group.add_argument(
        "--weight_decay", default=0.0, type=float, help="Optimizer weight decay."
    )
    group.add_argument(
        "--lr_anneal_steps",
        default=0,
        type=int,
        help="Number of learning rate anneal steps.",
    )
    group.add_argument(
        "--train_dataset_repeat_times",
        default=1000,
        type=int,
        help="Repeat the training dataset to save training time",
    )
    group.add_argument(
        "--eval_during_training",
        action="store_true",
        help="If True, will run evaluation during training.",
    )
    group.add_argument(
        "--log_interval", default=100, type=int, help="Log losses each N steps"
    )
    group.add_argument(
        "--save_interval",
        default=5000,
        type=int,
        help="Save checkpoints and run evaluation each N steps",
    )
    group.add_argument(
        "--num_steps",
        default=6000000,
        type=int,
        help="Training will stop after the specified number of steps.",
    )
    group.add_argument(
        "--resume_checkpoint",
        default="",
        type=str,
        help="If not empty, will start from the specified checkpoint (path to model###.pt file).",
    )
    group.add_argument(
        "--load_optimizer",
        action="store_true",
        help="If True, will also load the saved optimizer state for network initialization",
    )
    group.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of dataloader workers.",
    )


def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--overlapping_test",
        action="store_true",
        help="enabling overlapping test",
    )
    group.add_argument(
        "--num_per_batch",
        default=256,
        type=int,
        help="the batch size of each split during non-overlapping testing",
    )
    group.add_argument(
        "--sld_wind_size",
        default=70,
        type=int,
        help="the sliding window size",
    )
    group.add_argument(
        "--vis",
        action="store_true",
        help="visualize the output",
    )
    group.add_argument(
        "--fix_noise",
        action="store_true",
        help="fix init noise for the output",
    )
    group.add_argument(
        "--fps",
        default=30,
        type=int,
        help="FPS",
    )
    group.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Path to results dir (auto created by the script). "
        "If empty, will create dir in parallel to checkpoint.",
    )
    group.add_argument(
        "--support_dir",
        type=str,
        help="the dir that you store your smplh and dmpls dirs",
    )


def add_evaluation_options(parser):
    group = parser.add_argument_group("eval")
    group.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def sample_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)
