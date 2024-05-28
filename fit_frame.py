from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import smplx
import torch
import torch.optim as optim
import trimesh
from tqdm import tqdm

from camera import Camera
from loss import SITHLoss
from optimizer import create_closure, create_optimizer
from prior import MaxMixturePrior
from utils import (
    batch_rodrigues,
    estimate_initial_t,
    rotation_matrix_to_angle_axis,
    so3_exp_map,
    so3_log_map,
)
from visualization import visualize, visualize_paired

LR = 3e-3
LR_ORIENT = 1e-3
LR_BETAS = 1e-3
LR_POSE = 3e-3


def fit_frame(
    body_model: smplx.SMPLX,
    keypoints: np.ndarray,
    img: np.ndarray,
    camera: Camera,
    img_output_path: Path,
    mesh_output_path: Path,
    optimizer_type: Literal["ADAM", "LBFGS"] = "LBFGS",
    opt_cam_rot: bool = False,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    keypoints = torch.tensor(keypoints).to(device)
    gt_joints = keypoints[:, :2]

    """ Initial SMPL-X parameters"""

    pose_prior = MaxMixturePrior().to(device)
    param_body_pose = pose_prior.get_mean().to(device)[:, :63]  # (1, 63)

    param_betas = (
        body_model.betas.clone().detach().reshape(1, -1).contiguous()
    )  # (1, 10)

    # Estimate the initial camera translation
    estimate_t = torch.tensor([0.0, 0.0, 2.5]).to(device)
    camera.update_translation(estimate_t.view_as(camera.translation).detach().clone())
    camera.rotation.to(device)

    if opt_cam_rot:
        # Optimize over the logarithm of the rotation, use exponential maps to obtain the rotation matrix
        # log_rotation = so3_log_map(camera.rotation.unsqueeze(0)).requires_grad_(True)
        opt_rotation = (
            torch.Tensor([[0, 0, 0]]).view(1, 3).to(device).requires_grad_(True)
        )
        body_model.global_orient.requires_grad = False
    else:
        orient_angle = batch_rodrigues(body_model.global_orient)  # [1, 3]
        p = torch.tensor(np.pi)
        c, s = torch.cos(p), torch.sin(p)
        Rx = torch.tensor([[1, 0, 0], [0, c, s], [0, -s, c]]).to(device)
        aa = Rx.T @ orient_angle

        param_global_orient = (
            rotation_matrix_to_angle_axis(aa).squeeze().detach().cpu().data
        )

        opt_rotation = torch.tensor(
            [[param_global_orient[0], param_global_orient[1], param_global_orient[2]]],
            device=device,
            requires_grad=True,
        )

    opt_betas = param_betas.requires_grad_(True)
    opt_body_pose = param_body_pose.reshape(1, -1, 3).requires_grad_(True)

    opt_params = [
        camera.translation,
        opt_rotation,
        opt_betas,
        opt_body_pose,
    ]

    optimizer, scheduler = create_optimizer(optimizer_type, opt_params)

    sith_loss = SITHLoss(keypoints).to(device)

    n_iter = 500 if optimizer_type == "ADAM" else 20

    for _ in tqdm(range(n_iter), desc="Optimization Loop"):

        if opt_cam_rot:

            camera.rotation.data = so3_exp_map(opt_rotation)[0].squeeze(0)
            camera.rotation.data[1, 1] *= -1
            camera.rotation.data[2, 2] *= -1

            global_orient = body_model.global_orient.clone().detach()

        else:
            global_orient = opt_rotation

        if optimizer_type == "ADAM":
            optimizer.zero_grad()

            body_model_output = body_model(
                global_orient=global_orient,
                betas=opt_betas,
                body_pose=opt_body_pose.view(1, -1),
            )

            model_keypoints = camera(body_model_output.joints.squeeze())

            loss = sith_loss(model_keypoints)
            loss.backward()

            optimizer.step()
            scheduler.step(loss)

            vertices = camera(body_model_output.vertices.squeeze())

            visualize(
                img,
                model_keypoints,
                gt_joints,
                vertices=vertices,
            )

        elif optimizer_type == "LBFGS":

            closure_fn = create_closure(
                body_model,
                opt_body_pose,
                opt_betas,
                global_orient,
                camera,
                sith_loss,
                optimizer,
                img,
                gt_joints,
                visualize,
            )

            optimizer.step(closure=closure_fn)
            scheduler.step()

    tqdm.write(f"Final rotation: {camera.rotation.data}")
    tqdm.write(f"Final log r: {opt_rotation}")

    # Save the final results

    body_model_output = body_model(
        global_orient=global_orient,
        betas=opt_betas,
        body_pose=opt_body_pose.view(1, -1),
    )

    model_keypoints = camera(body_model_output.joints.squeeze())
    vertices_2d = camera(body_model_output.vertices.squeeze())

    cv2.imwrite(
        img_output_path.as_posix(),
        255 * visualize(img, model_keypoints, gt_joints, vertices=vertices_2d),
    )

    # Save the mesh
    mesh = trimesh.Trimesh(
        vertices=body_model_output.vertices.squeeze().detach().cpu().numpy(),
        faces=body_model.faces,
    )
    mesh.export(mesh_output_path.as_posix())

    body_model.reset_params()
