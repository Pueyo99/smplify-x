import json
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class JointMapper(nn.Module):
    def __init__(self, joint_maps: np.ndarray = None) -> None:
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer(
                "joint_maps", torch.tensor(joint_maps, dtype=torch.long)
            )

    def forward(self, joints, **kargs) -> torch.Tensor:
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(
        self,
        data_folder,
        img_folder="images",
        keyp_folder="keypoints",
        use_hands=False,
        use_face=False,
        use_face_contour=False,
        joints_to_ign=None,
        openpose_format="coco25",
        dtype=torch.float32,
        **kwargs
    ) -> None:
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = self.NUM_BODY_JOINTS + 2 * self.NUM_HAND_JOINTS * use_hands

        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)

        self.img_paths = [
            osp.join(self.img_folder, img_name)
            for img_name in os.listdir(self.img_folder)
            if img_name.endswith(".png")
            or img_name.endswith(".jpg")
            or img_name.endswith(".jpeg")
            and not img_name.startswith(".")
        ]
        self.img_paths = sorted(self.img_paths)
        self.cnt = 0

    def get_model2data(self) -> np.ndarray:
        return smpl_to_openpose(
            use_hands=self.use_hands,
            use_face=self.use_face,
            use_face_contour=self.use_face_contour,
            openpose_format=self.openpose_format,
        )

    def get_left_shoulder(self) -> int:
        return 2

    def get_right_shoulder(self) -> int:
        return 5

    def get_joint_weights(self) -> torch.Tensor:
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(
            self.num_joints
            + 2 * self.use_hands
            + 51 * self.use_face
            + 17 * self.use_face_contour,
            dtype=np.float32,
        )

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.0
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path) -> dict:
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn = osp.split(img_path)[1]
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        keypoints_path = osp.join(self.keyp_folder, img_fn + "_keypoints.json")
        keypoints = read_keypoints(
            keypoints_path,
            use_hands=self.use_hands,
            use_face=self.use_face,
            use_face_contour=self.use_face_contour,
        )

        if len(keypoints) < 1:
            return {}
        keypoints = np.stack(keypoints)

        output_dict = {
            "img": img,
            "img_name": img_fn,
            "img_path": img_path,
            "keypoints": keypoints,
        }

        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)


def read_keypoints(
    keypoints_path, use_hands=True, use_face=True, use_face_contour=False
) -> list:
    with open(keypoints_path) as keypoints_file:
        data = json.load(keypoints_file)

    keypoints = []

    for idx, person_data in enumerate(data["people"]):
        body_keypoints = np.array(person_data["pose_keypoints_2d"], dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])

        if use_hands:
            left_hand_keyp = np.array(
                person_data["hand_left_keypoints_2d"], dtype=np.float32
            ).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data["hand_right_keypoints_2d"], dtype=np.float32
            ).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0
            )

        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data["face_keypoints_2d"], dtype=np.float32
            ).reshape([-1, 3])[17 : 17 + 51, :]

            if use_face_contour:
                contour_keyps = np.array(
                    person_data["face_keypoints_2d"], dtype=np.float32
                ).reshape([-1, 3])[:17, :]

            else:
                contour_keyps = np.array([], dtype=body_keypoints.dtype).reshape(0, 3)

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0
            )

        keypoints.append(body_keypoints)

    return keypoints


def smpl_to_openpose(
    use_hands=True,
    use_face=True,
    use_face_contour=False,
    openpose_format="coco25",
) -> np.ndarray:
    """Returns the indices of the permutation that maps OpenPose to SMPL

    Parameters
    ----------
    use_hands: bool, optional
        Flag for adding to the returned permutation the mapping for the
        hand keypoints. Defaults to True
    use_face: bool, optional
        Flag for adding to the returned permutation the mapping for the
        face keypoints. Defaults to True
    use_face_contour: bool, optional
        Flag for appending the facial contour keypoints. Defaults to False
    openpose_format: bool, optional
        The output format of OpenPose. For now only COCO-25 and COCO-19 is
        supported. Defaults to 'coco25'

    """
    if openpose_format.lower() == "coco25":
        # SMPLX
        body_mapping = np.array(
            [
                55,
                12,
                17,
                19,
                21,
                16,
                18,
                20,
                0,
                2,
                5,
                8,
                1,
                4,
                7,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
            ],
            dtype=np.int32,
        )
        mapping = [body_mapping]
        if use_hands:
            lhand_mapping = np.array(
                [
                    20,
                    37,
                    38,
                    39,
                    66,
                    25,
                    26,
                    27,
                    67,
                    28,
                    29,
                    30,
                    68,
                    34,
                    35,
                    36,
                    69,
                    31,
                    32,
                    33,
                    70,
                ],
                dtype=np.int32,
            )
            rhand_mapping = np.array(
                [
                    21,
                    52,
                    53,
                    54,
                    71,
                    40,
                    41,
                    42,
                    72,
                    43,
                    44,
                    45,
                    73,
                    49,
                    50,
                    51,
                    74,
                    46,
                    47,
                    48,
                    75,
                ],
                dtype=np.int32,
            )

            mapping += [lhand_mapping, rhand_mapping]
        if use_face:
            #  end_idx = 127 + 17 * use_face_contour
            face_mapping = np.arange(76, 127 + 17 * use_face_contour, dtype=np.int32)
            mapping += [face_mapping]

        return np.concatenate(mapping)
    elif openpose_format == "coco19":
        body_mapping = np.array(
            [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59],
            dtype=np.int32,
        )
        mapping = [body_mapping]
        if use_hands:
            lhand_mapping = np.array(
                [
                    20,
                    37,
                    38,
                    39,
                    60,
                    25,
                    26,
                    27,
                    61,
                    28,
                    29,
                    30,
                    62,
                    34,
                    35,
                    36,
                    63,
                    31,
                    32,
                    33,
                    64,
                ],
                dtype=np.int32,
            )
            rhand_mapping = np.array(
                [
                    21,
                    52,
                    53,
                    54,
                    65,
                    40,
                    41,
                    42,
                    66,
                    43,
                    44,
                    45,
                    67,
                    49,
                    50,
                    51,
                    68,
                    46,
                    47,
                    48,
                    69,
                ],
                dtype=np.int32,
            )

            mapping += [lhand_mapping, rhand_mapping]
        if use_face:
            face_mapping = np.arange(
                70, 70 + 51 + 17 * use_face_contour, dtype=np.int32
            )
            mapping += [face_mapping]

        return np.concatenate(mapping)
    else:
        raise ValueError("Unknown joint format: {}".format(openpose_format))
