import torch
import torch.nn as nn


class KeypointLoss(nn.Module):
    def __init__(self, dtype=torch.float32, **kwargs):
        super(KeypointLoss, self).__init__()
        self.dtype = dtype

    def forward(self, model_joints, gt_joints):

        reprojection_loss = torch.sum((model_joints - gt_joints) ** 2)

        return reprojection_loss


class CameraLoss(nn.Module):
    def __init__(
        self,
        estimate_t: torch.Tensor,
        reprojection_weight: float = 0.01,
        depth_weight: float = 100,
        dtype=torch.float32,
        **kwargs
    ):
        super(CameraLoss, self).__init__()
        self.dtype = dtype
        self.estimate_t = estimate_t
        self.reprojection_weight = reprojection_weight
        self.depth_weight = depth_weight

        # Select the joints used for the reprojection loss
        self.joint_idx = torch.arange(24)

    def forward(self, model_joints, gt_joints, camera):

        reprojection_loss = self.reprojection_weight * torch.sum(
            (model_joints[self.joint_idx, :] - gt_joints[self.joint_idx, :]) ** 2
        )

        # Depth loss is used to avoid the camera translation
        # from deviating from the original estimation
        depth_loss = (self.depth_weight**2) * torch.sum(
            (camera.translation[2] - self.estimate_t[2]) ** 2
        )

        return reprojection_loss + depth_loss


class CameraInitLoss(nn.Module):

    def __init__(
        self,
        init_joints_idxs,
        trans_estimation=None,
        data_weight=1.0,
        depth_loss_weight=1e2,
        dtype=torch.float32,
        **kwargs
    ):
        super(CameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                "trans_estimation", to_tensor(trans_estimation, dtype=dtype)
            )
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer("data_weight", torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            "init_joints_idxs", to_tensor(init_joints_idxs.squeeze(), dtype=torch.long)
        )
        self.register_buffer(
            "depth_loss_weight", torch.tensor(depth_loss_weight, dtype=dtype)
        )

    def reset_loss_weights(self, loss_weight: dict) -> None:
        for key in loss_weight:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(
                    loss_weight[key],
                    dtype=weight_tensor.dtype,
                    device=weight_tensor.device,
                )
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            gt_joints.index_select(1, self.init_joints_idxs)
            - projected_joints.index_select(1, self.init_joints_idxs),
            2,
        )
        joint_loss = torch.sum(joint_error) * self.data_weight**2

        depth_loss = 0.0
        if self.depth_loss_weight.item() > 0 and self.trans_estimation is not None:
            depth_loss = self.depth_loss_weight**2 * torch.sum(
                (camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2)
            )

        return joint_loss + depth_loss


class SMPLifyLoss(nn.Module):
    def __init__(
        self,
        rho: int,
        body_pose_prior: nn.Module,
        shape_prior: nn.Module,
        angle_prior: nn.Module,
        jaw_prior: nn.Module,
        left_hand_prior: nn.Module,
        right_hand_prior: nn.Module,
        expression_prior: nn.Module,
        data_weight: float = 1.0,
        body_pose_weight: float = 0.0,
        shape_weight: float = 0.0,
        bending_prior_weight: float = 0.0,
        hand_prior_weight: float = 0.0,
        jaw_prior_weight: float = 0.0,
        expression_prior_weight: float = 0.0,
        dtype=torch.float32,
        **kwargs
    ) -> None:

        super(SMPLifyLoss, self).__init__()

        self.rho = rho

        self.body_pose_prior = body_pose_prior
        self.shape_prior = shape_prior
        self.angle_prior = angle_prior

        self.jaw_prior = jaw_prior
        self.left_hand_prior = left_hand_prior
        self.right_hand_prior = right_hand_prior
        self.expression_prior = expression_prior

        self.register_buffer("data_weight", torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            "body_pose_weight", torch.tensor(body_pose_weight, dtype=dtype)
        )
        self.register_buffer("shape_weight", torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer(
            "bending_prior_weight", torch.tensor(bending_prior_weight, dtype=dtype)
        )

        self.register_buffer(
            "hand_prior_weight",
            torch.tensor(hand_prior_weight, dtype=dtype),
        )

        self.register_buffer(
            "jaw_prior_weights", torch.tensor(jaw_prior_weight, dtype=dtype)
        )

        self.register_buffer(
            "expression_prior_weight",
            torch.tensor(expression_prior_weight, dtype=dtype),
        )

    def reset_loss_weights(self, loss_weight: dict) -> None:
        for key in loss_weight:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(
                    loss_weight[key],
                    dtype=weight_tensor.dtype,
                    device=weight_tensor.device,
                )
                setattr(self, key, weight_tensor)

    def compute_joint_distance(self, gt_joints, projected_joints):
        squared_res = (gt_joints - projected_joints) ** 2
        squared_rho = self.rho

        return torch.div(squared_res, squared_res + squared_rho) * squared_rho

    def forward(
        self, body_model_output, camera, gt_joints, joint_conf, joint_weights, **kwargs
    ):

        # Compute the joint loss
        projected_joints = camera(body_model_output.joints[:, :, :])
        joint_weights = (joint_weights * joint_conf).unsqueeze(-1)

        joint_dist = self.compute_joint_distance(gt_joints, projected_joints)
        joint_loss = torch.sum(joint_weights**2 * joint_dist) * self.data_weight**2

        # Compute the shape loss
        shape_loss = (
            torch.sum(self.shape_prior(body_model_output.betas)) * self.shape_weight**2
        )

        # Compute the angle prior over the joint rotations. Used to prevent extreme rotation
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_loss = torch.sum(self.angle_prior(body_pose)) * self.bending_prior_weight

        # Hand prior loss
        left_hand_prior_loss = (
            torch.sum(self.left_hand_prior(body_model_output.left_hand_pose))
            * self.hand_prior_weight**2
        )

        right_hand_prior_loss = (
            torch.sum(self.right_hand_prior(body_model_output.right_hand_pose))
            * self.hand_prior_weight**2
        )

        jaw_prior_loss = torch.sum(
            self.jaw_prior(body_model_output.jaw_pose.mul(self.jaw_prior_weights))
        )

        expression_loss = (
            torch.sum(self.expression_prior(body_model_output.expression))
            * self.expression_prior_weight**2
        )

        total_loss = (
            joint_loss
            + shape_loss
            + angle_loss
            + left_hand_prior_loss
            + right_hand_prior_loss
            + jaw_prior_loss
            + expression_loss
        )

        return total_loss


def to_tensor(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if type(tensor) == torch.Tensor:
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype=dtype)


class SITHLoss(nn.Module):
    def __init__(self, gt_keypoints: torch.Tensor, dtype=torch.float32, **kwargs):
        super(SITHLoss, self).__init__()

        self.body_loss_weight = 20.0
        self.hand_loss_weight = 50.0
        self.face_loss_weight = 20.0
        self.leg_loss_weight = 20.0

        self.body_idx = [1, 2, 5, 8, 9, 12]
        self.leg_idx = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
        self.lhand_idx = [5, 6, 7, 29, 33, 37, 41, 45]
        self.rhand_idx = [2, 3, 4, 50, 54, 58, 62, 66]
        self.face_idx = [0, 15, 16, 17, 18, 107, 110, 113, 116]

        self.gt_keypoints = gt_keypoints

        self.body_kps = gt_keypoints[self.body_idx, :2]
        self.conf_body_kps = gt_keypoints[self.body_idx, 2]

        self.leg_kps = gt_keypoints[self.leg_idx, :2]
        self.conf_leg_kps = gt_keypoints[self.leg_idx, 2]

        self.lhand_kps = gt_keypoints[self.lhand_idx, :2]
        self.conf_lhand_kps = gt_keypoints[self.lhand_idx, 2]

        self.rhand_kps = gt_keypoints[self.rhand_idx, :2]
        self.conf_rhand_kps = gt_keypoints[self.rhand_idx, 2]

        self.face_kps = gt_keypoints[self.face_idx, :2]
        self.conf_face_kps = gt_keypoints[self.face_idx, 2]

    def forward(self, model_keypoints: torch.Tensor) -> torch.Tensor:
        body_loss = (
            torch.norm(self.body_kps - model_keypoints[self.body_idx, :2], dim=1)
            * self.conf_body_kps
        ).mean(dim=0) * self.body_loss_weight

        lhand_loss = (
            torch.norm(self.lhand_kps - model_keypoints[self.lhand_idx, :2], dim=1)
            * self.conf_lhand_kps
        ).mean(dim=0) * self.hand_loss_weight

        rhand_loss = (
            torch.norm(self.rhand_kps - model_keypoints[self.rhand_idx, :2], dim=1)
            * self.conf_rhand_kps
        ).mean(dim=0) * self.hand_loss_weight

        face_loss = (
            torch.norm(self.face_kps - model_keypoints[self.face_idx, :2], dim=1)
            * self.conf_face_kps
        ).mean(dim=0) * self.face_loss_weight

        leg_loss = (
            torch.norm(self.leg_kps - model_keypoints[self.leg_idx, :2], dim=1)
            * self.conf_leg_kps
        ).mean(dim=0) * self.leg_loss_weight

        return body_loss + lhand_loss + rhand_loss + face_loss + leg_loss
