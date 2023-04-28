# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import math
import os
import random

import numpy as np

import torch

from data_loaders.dataloader import load_data, TestDataset

from human_body_prior.body_model.body_model import BodyModel as BM

from model.networks import PureMLP
from tqdm import tqdm

from utils import utils_transform, utils_visualize
from utils.metrics import get_metric_function
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import sample_args

device = torch.device("cuda")

#####################
RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre",
    "mpjpe",
    "mpjve",
    "handpe",
    "upperpe",
    "lowerpe",
    "rootpe",
    "pred_jitter",
]
gt_metrics = [
    "gt_jitter",
]
all_metrics = pred_metrics + gt_metrics

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)  # 57.2958 grads
metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
    "gt_mpjpe": METERS_TO_CENTIMETERS,
    "gt_mpjve": METERS_TO_CENTIMETERS,
    "gt_handpe": METERS_TO_CENTIMETERS,
    "gt_rootpe": METERS_TO_CENTIMETERS,
    "gt_upperpe": METERS_TO_CENTIMETERS,
    "gt_lowerpe": METERS_TO_CENTIMETERS,
}

#####################


class BodyModel(torch.nn.Module):
    def __init__(self, support_dir):
        super().__init__()

        device = torch.device("cuda")
        subject_gender = "male"
        bm_fname = os.path.join(
            support_dir, "smplh/{}/model.npz".format(subject_gender)
        )
        dmpl_fname = os.path.join(
            support_dir, "dmpls/{}/model.npz".format(subject_gender)
        )
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        body_model = BM(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        ).to(device)
        self.body_model = body_model.eval()

    def forward(self, body_params):
        with torch.no_grad():
            body_pose = self.body_model(
                **{
                    k: v
                    for k, v in body_params.items()
                    if k in ["pose_body", "trans", "root_orient"]
                }
            )
        return body_pose


def non_overlapping_test(
    args,
    data,
    sample_fn,
    dataset,
    model,
    num_per_batch=256,
    model_type="mlp",
):
    gt_data, sparse_original, body_param, head_motion, filename = (
        data[0],
        data[1],
        data[2],
        data[3],
        data[4],
    )
    gt_data = gt_data.cuda().float()
    sparse_original = sparse_original.cuda().float()
    head_motion = head_motion.cuda().float()
    num_frames = head_motion.shape[0]

    output_samples = []
    count = 0
    sparse_splits = []
    flag_index = None

    if args.input_motion_length <= num_frames:
        while count < num_frames:
            if count + args.input_motion_length > num_frames:
                tmp_k = num_frames - args.input_motion_length
                sub_sparse = sparse_original[
                    :, tmp_k : tmp_k + args.input_motion_length
                ]
                flag_index = count - tmp_k
            else:
                sub_sparse = sparse_original[
                    :, count : count + args.input_motion_length
                ]
            sparse_splits.append(sub_sparse)
            count += args.input_motion_length
    else:
        flag_index = args.input_motion_length - num_frames
        tmp_init = sparse_original[:, :1].repeat(1, flag_index, 1).clone()
        sub_sparse = torch.concat([tmp_init, sparse_original], dim=1)
        sparse_splits = [sub_sparse]

    n_steps = len(sparse_splits) // num_per_batch
    if len(sparse_splits) % num_per_batch > 0:
        n_steps += 1
    # Split the sequence into n_steps non-overlapping batches

    if args.fix_noise:
        # fix noise seed for every frame
        noise = torch.randn(1, 1, 1).cuda()
        noise = noise.repeat(1, args.input_motion_length, args.motion_nfeat)
    else:
        noise = None

    for step_index in range(n_steps):
        sparse_per_batch = torch.cat(
            sparse_splits[
                step_index * num_per_batch : (step_index + 1) * num_per_batch
            ],
            dim=0,
        )

        new_batch_size = sparse_per_batch.shape[0]

        if model_type == "diffusion":
            sample = sample_fn(
                model,
                (new_batch_size, args.input_motion_length, args.motion_nfeat),
                sparse=sparse_per_batch,
                clip_denoised=False,
                model_kwargs=None,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=noise,
                const_noise=False,
            )
        elif model_type == "mlp":
            sample = model(sparse_per_batch)

        if flag_index is not None and step_index == n_steps - 1:
            last_batch = sample[-1]
            last_batch = last_batch[flag_index:]
            sample = sample[:-1].reshape(-1, args.motion_nfeat)
            sample = torch.cat([sample, last_batch], dim=0)
        else:
            sample = sample.reshape(-1, args.motion_nfeat)

        if not args.no_normalization:
            output_samples.append(dataset.inv_transform(sample.cpu().float()))
        else:
            output_samples.append(sample.cpu().float())

    return output_samples, body_param, head_motion, filename


def overlapping_test(
    args,
    data,
    sample_fn,
    dataset,
    model,
    sld_wind_size=70,
    model_type="diffusion",
):
    assert (
        model_type == "diffusion"
    ), "currently only diffusion model supports overlapping test!!!"

    gt_data, sparse_original, body_param, head_motion, filename = (
        data[0],
        data[1],
        data[2],
        data[3],
        data[4],
    )
    gt_data = gt_data.cuda().float()
    sparse_original = sparse_original.cuda().float()
    head_motion = head_motion.cuda().float()
    num_frames = head_motion.shape[0]

    output_samples = []
    count = 0
    sparse_splits = []
    flag_index = None

    if num_frames < args.input_motion_length:
        flag_index = args.input_motion_length - num_frames
        tmp_init = sparse_original[:, :1].repeat(1, flag_index, 1).clone()
        sub_sparse = torch.concat([tmp_init, sparse_original], dim=1)
        sparse_splits = [sub_sparse]

    else:
        while count + args.input_motion_length <= num_frames:
            if count == 0:
                sub_sparse = sparse_original[
                    :, count : count + args.input_motion_length
                ]
                tmp_idx = 0
            else:
                sub_sparse = sparse_original[
                    :, count : count + args.input_motion_length
                ]
                tmp_idx = args.input_motion_length - sld_wind_size
            sparse_splits.append([sub_sparse, tmp_idx])
            count += sld_wind_size

        if count < num_frames:
            sub_sparse = sparse_original[:, -args.input_motion_length :]
            tmp_idx = args.input_motion_length - (
                num_frames - (count - sld_wind_size + args.input_motion_length)
            )
            sparse_splits.append([sub_sparse, tmp_idx])

    memory = None  # init memory

    if args.fix_noise:
        # fix noise seed for every frame
        noise = torch.randn(1, 1, 1).cuda()
        noise = noise.repeat(1, args.input_motion_length, args.motion_nfeat)
    else:
        noise = None

    for step_index in range(len(sparse_splits)):
        sparse_per_batch = sparse_splits[step_index][0]
        memory_end_index = sparse_splits[step_index][1]

        new_batch_size = sparse_per_batch.shape[0]
        assert new_batch_size == 1

        if memory is not None:
            model_kwargs = {}
            model_kwargs["y"] = {}
            model_kwargs["y"]["inpainting_mask"] = torch.zeros(
                (
                    new_batch_size,
                    args.input_motion_length,
                    args.motion_nfeat,
                )
            ).cuda()
            model_kwargs["y"]["inpainting_mask"][:, :memory_end_index, :] = 1
            model_kwargs["y"]["inpainted_motion"] = torch.zeros(
                (
                    new_batch_size,
                    args.input_motion_length,
                    args.motion_nfeat,
                )
            ).cuda()
            model_kwargs["y"]["inpainted_motion"][:, :memory_end_index, :] = memory[
                :, -memory_end_index:, :
            ]
        else:
            model_kwargs = None

        sample = sample_fn(
            model,
            (new_batch_size, args.input_motion_length, args.motion_nfeat),
            sparse=sparse_per_batch,
            clip_denoised=False,
            model_kwargs=None,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=noise,
            const_noise=False,
        )

        memory = sample.clone().detach()

        if flag_index is not None:
            sample = sample[:, flag_index:].cpu().reshape(-1, args.motion_nfeat)
        else:
            sample = sample[:, memory_end_index:].reshape(-1, args.motion_nfeat)

        if not args.no_normalization:
            output_samples.append(dataset.inv_transform(sample.cpu().float()))
        else:
            output_samples.append(sample.cpu().float())

    return output_samples, body_param, head_motion, filename


def evaluate_prediction(
    args,
    metrics,
    sample,
    body_model,
    sample_index,
    head_motion,
    body_param,
    fps,
    filename,
):
    motion_pred = sample.squeeze().cuda()
    # Get the  prediction from the model
    model_rot_input = (
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach())
        .reshape(motion_pred.shape[0], -1)
        .float()
    )

    T_head2world = head_motion.clone().cuda()
    t_head2world = T_head2world[:, :3, 3].clone()

    # Get the offset between the head and other joints using forward kinematic model
    body_pose_local = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
        }
    ).Jtr

    # Get the offset in global coordiante system between head and body_world.
    t_head2root = -body_pose_local[:, 15, :]
    t_root2world = t_head2root + t_head2world.cuda()

    predicted_body = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            "trans": t_root2world,
        }
    )
    predicted_position = predicted_body.Jtr[:, :22, :]

    # Get the predicted position and rotation
    predicted_angle = model_rot_input

    for k, v in body_param.items():
        body_param[k] = v.squeeze().cuda()
        body_param[k] = body_param[k][-predicted_angle.shape[0] :, ...]

    # Get the  ground truth position from the model
    gt_body = body_model(body_param)
    gt_position = gt_body.Jtr[:, :22, :]

    # Create animation
    if args.vis:
        video_dir = args.output_dir
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        save_filename = filename.split(".")[0].replace("/", "-")
        save_video_path = os.path.join(video_dir, save_filename + ".mp4")
        utils_visualize.save_animation(
            body_pose=predicted_body,
            savepath=save_video_path,
            bm=body_model.body_model,
            fps=fps,
            resolution=(800, 800),
        )
        save_video_path_gt = os.path.join(video_dir, save_filename + "_gt.mp4")
        if not os.path.exists(save_video_path_gt):
            utils_visualize.save_animation(
                body_pose=gt_body,
                savepath=save_video_path_gt,
                bm=body_model.body_model,
                fps=fps,
                resolution=(800, 800),
            )

    gt_angle = body_param["pose_body"]
    gt_root_angle = body_param["root_orient"]

    predicted_root_angle = predicted_angle[:, :3]
    predicted_angle = predicted_angle[:, 3:]

    upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    eval_log = {}
    for metric in metrics:
        eval_log[metric] = (
            get_metric_function(metric)(
                predicted_position,
                predicted_angle,
                predicted_root_angle,
                gt_position,
                gt_angle,
                gt_root_angle,
                upper_index,
                lower_index,
                fps,
            )
            .cpu()
            .numpy()
        )

    torch.cuda.empty_cache()
    return eval_log


def load_diffusion_model(args):
    print("Creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]
    model, diffusion = create_model_and_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model.to("cuda:0")  # dist_util.dev())
    model.eval()  # disable random masking
    return model, diffusion


def load_mlp_model(args):
    model = PureMLP(
        args.latent_dim,
        args.input_motion_length,
        args.layers,
        args.sparse_dim,
        args.motion_nfeat,
    )
    model.eval()
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cuda:0")
    return model, None


def main():
    args = sample_args()

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fps = 60  # AMASS dataset requires 60 frames per second

    body_model = BodyModel(args.support_dir)
    print("Loading dataset...")
    filename_list, all_info, mean, std = load_data(
        args.dataset,
        args.dataset_path,
        "test",
    )
    dataset = TestDataset(
        args.dataset,
        mean,
        std,
        all_info,
        filename_list,
    )

    log = {}
    for metric in all_metrics:
        log[metric] = 0

    model_type = args.arch.split("_")[0]
    if model_type == "diffusion":
        model, diffusion = load_diffusion_model(args)
        sample_fn = diffusion.p_sample_loop
    elif model_type == "mlp":
        model, _ = load_mlp_model(args)
        sample_fn = None
    else:
        raise ValueError(f"Unknown model type {model_type}")

    if not args.overlapping_test:
        test_func = non_overlapping_test
        # batch size in the case of non-overlapping testing
        n_testframe = args.num_per_batch
    else:
        print("Overlapping testing...")
        test_func = overlapping_test
        # sliding window size in case of overlapping testing
        n_testframe = args.sld_wind_size

    for sample_index in tqdm(range(len(dataset))):

        output, body_param, head_motion, filename = test_func(
            args,
            dataset[sample_index],
            sample_fn,
            dataset,
            model,
            n_testframe,
            model_type=model_type,
        )

        sample = torch.cat(output, axis=0)

        instance_log = evaluate_prediction(
            args,
            all_metrics,
            sample,
            body_model,
            sample_index,
            head_motion,
            body_param,
            fps,
            filename,
        )
        for key in instance_log:
            log[key] += instance_log[key]

    # Print the value for all the metrics
    print("Metrics for the predictions")
    for metric in pred_metrics:
        print(log[metric] / len(dataset) * metrics_coeffs[metric])
    print("Metrics for the ground truth")
    for metric in gt_metrics:
        print(metric, log[metric] / len(dataset) * metrics_coeffs[metric])


if __name__ == "__main__":
    main()
