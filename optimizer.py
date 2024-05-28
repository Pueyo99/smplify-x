from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import LBFGS


def create_optimizer(
    type: Literal["ADAM", "LBFGS"],
    params: list,
    lr: float = 6e-3,
) -> tuple:

    if type == "ADAM":

        LR = 3e-3
        LR_ORIENT = 1e-3
        LR_BETAS = 1e-3
        LR_POSE = 3e-3

        opt_params = [
            {"params": params[0], "lr": LR},
            {"params": params[1], "lr": LR},
            {"params": params[2], "lr": LR_BETAS},
            {"params": params[3], "lr": LR_POSE},
        ]

        optimizer = optim.Adam(opt_params, betas=(0.9, 0.999), amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, verbose=0, min_lr=1e-5
        )

        return optimizer, scheduler

    elif type == "LBFGS":

        optimizer = torch.optim.LBFGS(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        return optimizer, scheduler


def create_closure(
    body_model: nn.Module,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    camera: nn.Module,
    sith_loss: nn.Module,
    optimizer: optim.Optimizer,
    img: torch.Tensor,
    gt_joints: torch.Tensor,
    visualize_fn: callable,
):
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()

        body_model_output = body_model(
            global_orient=global_orient,
            betas=betas,
            body_pose=body_pose.view(1, -1),
        )

        model_keypoints = camera(body_model_output.joints.squeeze())

        loss = sith_loss(model_keypoints)

        if loss.requires_grad:
            loss.backward()

        vertices = camera(body_model_output.vertices.squeeze())

        visualize_fn(
            img,
            model_keypoints,
            gt_joints,
            vertices=vertices,
        )

        return loss

    return closure
