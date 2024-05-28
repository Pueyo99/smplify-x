import torch
import torch.nn as nn
import torch.nn.functional as F


class Camera(nn.Module):

    def __init__(
        self,
        focal_length_x: float = 5000.0,
        focal_length_y: float = 5000.0,
        optical_center_x: float = 0.0,
        optical_center_y: float = 0.0,
        rotation: torch.Tensor = torch.eye(3),
        translation: torch.Tensor = torch.zeros((3,)),
    ):
        super(Camera, self).__init__()

        K = torch.zeros((3, 3), dtype=torch.float32)
        K[0, 0] = focal_length_x
        K[1, 1] = focal_length_y
        K[0, 2] = optical_center_x
        K[1, 2] = optical_center_y
        K[2, 2] = 1

        self.register_buffer("K", K)

        self.rotation = nn.Parameter(rotation)
        self.translation = nn.Parameter(translation)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Project 3D points to 2D using the camera model

        Args:
            points (torch.Tensor): Nx3 array of 3D points

        Returns:
            pixels (torch.Tensor): Nx2 array of 2D pixels
        """
        E = torch.concat([self.rotation, self.translation.unsqueeze(-1)], dim=1)  # 3x4

        points_h = torch.cat([points, torch.ones_like(points[:, 0:1])], dim=1)  # Nx4

        assert points_h.shape == (points.shape[0], 4)

        projected_points = torch.einsum("ji, ni -> nj", E, points_h)  # Nx3

        assert projected_points.shape == (points.shape[0], 3)

        pixels = torch.einsum("ji, ni -> nj", self.K, projected_points)

        pixels = pixels[:, :2] / pixels[:, 2:3]

        assert pixels.shape == (points.shape[0], 2)

        return pixels

    def update_center(self, x: float, y: float) -> None:
        """Update the optical center of the camera

        Args:
            x (float): New x coordinate
            y (float): New y coordinate
        """
        self.K[0, 2] = x
        self.K[1, 2] = y

    def get_center(self) -> tuple:
        """Get the optical center of the camera

        Returns:
            tuple: (x, y) coordinates of the optical center
        """
        return self.K[0, 2], self.K[1, 2]

    def update_focal_length(self, fx: float, fy: float) -> None:
        """Update the focal length of the camera

        Args:
            fx (float): New focal length in x direction
            fy (float): New focal length in y direction
        """
        self.K[0, 0] = fx
        self.K[1, 1] = fy

    def get_focal_length(self) -> float:
        """Get the focal length of the camera

        Returns:
            float: (focal_length_x, focal_length_y)
        """
        return self.K[0, 0], self.K[1, 1]

    def update_intrinsics(self, K: torch.Tensor) -> None:
        """Update the intrinsics of the camera

        Args:
            K (torch.Tensor): Bx3x3 array of camera matrices
        """
        self.K.data = K

    def update_translation(self, T: torch.Tensor) -> None:
        """Update the translation of the camera

        Args:
            T (torch.Tensor): Bx3 array of translation vectors
        """
        self.translation.data = T

    def update_rotation(self, R: torch.Tensor) -> None:
        """Update the rotation of the camera

        Args:
            R (torch.Tensor): Bx3x3 array of rotation matrices
        """
        self.rotation.data = R
