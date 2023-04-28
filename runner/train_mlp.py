# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import torch


def update_lr_multistep(
    nb_iter, total_iter, max_lr, min_lr, optimizer, lr_anneal_steps
):
    if nb_iter > lr_anneal_steps:
        current_lr = min_lr
    else:
        current_lr = max_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def train_step(
    motion_input,
    motion_target,
    model,
    optimizer,
    nb_iter,
    total_iter,
    max_lr,
    min_lr,
    device,
    lr_anneal_steps,
):

    motion_input = motion_input.to(device)
    motion_target = motion_target.to(device)

    motion_pred = model(motion_input)

    loss = torch.mean(
        torch.norm(
            (motion_pred - motion_target).reshape(-1, 6),
            2,
            1,
        )
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(
        nb_iter, total_iter, max_lr, min_lr, optimizer, lr_anneal_steps
    )

    return loss.item(), optimizer, current_lr
