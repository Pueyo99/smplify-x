import pickle
import warnings
from pathlib import Path

import smplx
import torch
import torch.nn as nn
from tqdm import tqdm

from camera import Camera
from dataset import JointMapper, OpenPose
from fit_frame import fit_frame

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_FOLDER: str = r"C:\Users\pueyo\smplify-x\models"
DATA_FOLDER: str = r"C:\Users\pueyo\smplify-x\data\albert"
dtype = torch.float32

output_folder = Path(r"C:\Users\pueyo\smplify-x\output")
result_folder = output_folder.joinpath("results")
mesh_folder = output_folder.joinpath("meshes")
out_img_folder = output_folder.joinpath("images")


def read_camera_parameters(path: str) -> dict:
    with open(path, "rb") as f:
        camera_params = pickle.load(f)

        return camera_params


if __name__ == "__main__":

    dataset = OpenPose(data_folder=DATA_FOLDER, use_hands=True, use_face=True)
    joint_mapper = JointMapper(dataset.get_model2data())

    model_params = dict(
        model_path=MODEL_FOLDER,
        gender="male",
        model_type="smplx",
        joint_mapper=joint_mapper,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=dtype,
    )

    model = smplx.create(**model_params)

    camera_params = read_camera_parameters("data/avatarrex/camera_parameters.pkl")
    camera_params = {d["img_name"]: d for d in camera_params}

    camera = Camera(
        rotation=torch.eye(3, dtype=dtype),
        translation=torch.zeros(3, dtype=dtype),
        focal_length_x=1250.0,
        focal_length_y=1250.0,
        optical_center_x=0.0,
        optical_center_y=0.0,
    )

    # Move everything to the same device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device=device)
    camera = camera.to(device=device)

    # Start the loop for every image in the dataset
    for idx, data in enumerate(
        tqdm(
            dataset,
            total=len(dataset),
            desc=f"Fitting images in {DATA_FOLDER}",
            leave=False,
        )
    ):

        img = data["img"]
        img_name = data["img_name"]
        keypoints = data["keypoints"]

        height, width, _ = img.shape
        camera.update_center(width / 2, height / 2)

        # Initialize camera instrinsics and extrinsics using the known camera parameters
        # cp = camera_params[f"{img_name}.jpg"]
        # camera.update_intrinsics(torch.Tensor(cp["K"]).to(device=device))
        # camera.update_rotation(torch.Tensor(cp["R"]).to(device=device))
        # camera.update_translation(torch.Tensor(cp["T"]).to(device=device))

        # For starters assume that there is only one person in the image
        keypoints = keypoints[[0]].squeeze(0)

        # Assign the proper paths to store the output
        current_result_path = result_folder.joinpath(f"{img_name}.pkl")
        current_mesh_path = mesh_folder.joinpath(f"{img_name}.obj")
        current_img_path = out_img_folder.joinpath(f"{img_name}.png")

        fit_frame(
            body_model=model,
            keypoints=keypoints,
            img=img,
            camera=camera,
            img_output_path=current_img_path,
            mesh_output_path=current_mesh_path,
        )
        break
