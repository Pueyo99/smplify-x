import cv2
import numpy as np


def visualize(img, model_keypoints, gt_joints, vertices=None):

    out_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    # Draw the OpenPose keypoints in 2D, directly on the image
    gt_points = gt_joints.squeeze().detach().cpu().numpy()
    out_img = write_points(out_img, gt_points, (0, 0, 255))

    # Draw the SMPL-X keypoints in 2D, directly on the images
    model_points = model_keypoints.squeeze().detach().cpu().numpy()
    out_img = write_points(out_img, model_points, (0, 255, 0))

    # Draw manually the vertices
    if vertices is not None:
        v_points = vertices.squeeze().detach().cpu().numpy()
        out_img = write_points(out_img, v_points, (255.0, 0, 0), size=1)

    """ out_img = cv2.resize(
        out_img, (int(out_img.shape[1] / 2), int(out_img.shape[0] / 2))
    ) """
    cv2.imshow("Frame", out_img)
    key = cv2.waitKey(1)

    return out_img


def visualize_paired(img, model_keypoints, gt_joints, vertices=None):

    np.random.seed(99)
    colors = np.random.uniform(0, 255, size=(model_keypoints.shape[0]))
    out_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    for i in range(model_keypoints.shape[0]):

        b = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        r = np.random.randint(0, 255)

        j = np.random.choice(range(3))
        if j == 0:
            b = 0
        elif j == 1:
            g = 0
        else:
            r = 0

        cv2.circle(
            out_img,
            (int(model_keypoints[i, 0]), int(model_keypoints[i, 1])),
            3,
            (b, g, r),  # BGR
            -1,
        )

        cv2.circle(
            out_img,
            (int(gt_joints[i, 0]), int(gt_joints[i, 1])),
            3,
            (b, g, r),
            -1,  # BGR
        )

    # Draw manually the vertices
    if vertices is not None:
        v_points = vertices.squeeze().detach().cpu().numpy()
        out_img = write_points(out_img, v_points, (255.0, 0, 0), size=1)

    """ out_img = cv2.resize(
        out_img, (int(out_img.shape[1] / 2), int(out_img.shape[0] / 2))
    ) """
    cv2.imshow("Frame", out_img)
    key = cv2.waitKey(1)

    return out_img


def write_points(
    img: np.ndarray, points: np.ndarray, color: tuple, size: int = 3
) -> np.ndarray:
    for i in range(points.shape[0]):
        cv2.circle(img, (int(points[i, 0]), int(points[i, 1])), size, color, -1)

    return img
