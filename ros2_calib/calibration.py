# MIT License
#
# Copyright (c) 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
# Copyright (c) 2026 GRASP Lab, University of Pennsylvania (Added Scaramuzza OCamCalib Fisheye Support)
#
# Acknowledgement: Portions of the modifications in this file, including 
# the adaptation for dual-fisheye datasets and unstitched frame processing, 
# were developed with the assistance of AI (Google Gemini and ChatGPT).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def objective_function(params, points_3d, points_2d, K, ocam_model=None):
    rvec = params[:3]
    tvec = params[3:]

    # Pinhole model
    if ocam_model is None:
        points_proj, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
        points_proj = points_proj.reshape(-1, 2)
    # 2. OCamCalib (Scaramuzza) model
    else:
        # transform LiDAR points to camera frame
        R, _ = cv2.Rodrigues(rvec)
        pts_cam = (R @ points_3d.T).T + tvec
        
        # re-project using Ocam model
        points_proj = ocam_model.project_points(pts_cam)

    error = (points_proj - points_2d).ravel()
    return error


def calibrate(
    correspondences,
    K,
    pnp_method=cv2.SOLVEPNP_ITERATIVE,
    lsq_method="lm",
    lsq_verbose=2,
    ocam_model=None,
    initial_transform=None,
):
    print("---" + "-" * 10 + " Starting Calibration " + "-" * 10 + "---")
    if len(correspondences) < 4:
        print("Error: Need at least 4 correspondences for calibration.")
        return np.identity(4)

    points_3d = np.array([c[1] for c in correspondences], dtype=np.float32)
    points_2d = np.array([c[0] for c in correspondences], dtype=np.float32)

    # Use initial_transform if provided
    if initial_transform is not None:
        rvec_init, _ = cv2.Rodrigues(initial_transform[:3, :3])
        tvec_init = initial_transform[:3, 3]
        initial_params = np.concatenate((rvec_init.ravel(), tvec_init.ravel()))
    else:
        initial_params = np.zeros(6)
    
    # If using OCam (Scaramuzza fisheye) model, cv2.solvePnPRansac assumes a pinhole model 
    # and utilizes a dummy camera matrix K, making it mathematically invalid and prone to finding 
    # a garbage pose. Therefore, we skip RANSAC and directly refine the manual initial_transform.
    use_ransac = (pnp_method is not None) and (ocam_model is None)
    
    if use_ransac:
        print(f"Running RANSAC on {len(points_2d)} correspondences...")
        try:
            success, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(
                points_3d,
                points_2d,
                K,
                None, 
                iterationsCount=100,
                reprojectionError=8.0,
                flags=pnp_method,
            )
            if not success:
                print("RANSAC failed to find a solution. Using zeros/initial_transform as initial guess.")
            else:
                inlier_count = len(inliers) if inliers is not None else 0
                print(f"RANSAC found {inlier_count} inliers out of {len(points_2d)} points.")

                if inlier_count >= 4:
                    if ocam_model is None:
                        inlier_points_3d = points_3d[inliers.ravel()]
                        inlier_points_2d = points_2d[inliers.ravel()]
                        points_3d = inlier_points_3d
                        points_2d = inlier_points_2d
                    
                    initial_params = np.concatenate((rvec_ransac.ravel(), tvec_ransac.ravel()))
                else:
                    print("Not enough inliers found by RANSAC. Using all points.")

        except cv2.error as e:
            print(f"An OpenCV error occurred during RANSAC: {e}")
            print("Falling back to simple least squares with all points.")

    print(f"\nRefining pose using least squares optimization (Method: {lsq_method})...")
    if ocam_model is not None:
        print(">> Using OCamCalib Fisheye Projection Model for residuals computation.")

    res = least_squares(
        objective_function,
        initial_params,
        args=(points_3d, points_2d, K, ocam_model),
        method=lsq_method,
        verbose=lsq_verbose,
    )

    rvec_opt = res.x[:3]
    tvec_opt = res.x[3:]
    R_opt, _ = cv2.Rodrigues(rvec_opt)

    extrinsics = np.identity(4)
    extrinsics[:3, :3] = R_opt
    extrinsics[:3, 3] = tvec_opt.ravel()

    print("\n---" + "-" * 10 + " Calibration Finished " + "-" * 10 + "---")
    rpy = Rotation.from_matrix(R_opt).as_euler("xyz", degrees=True)
    print(f"Translation (x, y, z): {tvec_opt[0]:.4f}, {tvec_opt[1]:.4f}, {tvec_opt[2]:.4f}")
    print(f"Rotation (roll, pitch, yaw): {rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}")
    print("Final Extrinsic Matrix:")
    print(extrinsics)

    return extrinsics


def calibrate_dual_lidar(
    master_cam_correspondences,
    lidar_lidar_correspondences,
    K,
    pnp_method=cv2.SOLVEPNP_ITERATIVE,
    lsq_method="lm",
    lsq_verbose=2,
    ocam_model=None, 
    initial_transform=None,
):
    print("---" + "-" * 10 + " Starting Dual LiDAR Calibration " + "-" * 10 + "---")

    print("Step 1: Calibrating master LiDAR to camera...")
    master_to_camera = calibrate(
        master_cam_correspondences, K, pnp_method, lsq_method, lsq_verbose, ocam_model, initial_transform
    )

    if len(lidar_lidar_correspondences) < 3:
        print("Error: Need at least 3 LiDAR-to-LiDAR correspondences.")
        return master_to_camera, np.eye(4)

    print(f"Step 2: Calibrating second LiDAR to master LiDAR using {len(lidar_lidar_correspondences)} correspondences...")

    second_lidar_points = []
    master_lidar_points = []

    for second_3d, corr_data in lidar_lidar_correspondences.items():
        second_lidar_points.append(list(second_3d))
        master_lidar_points.append(corr_data["master_3d_mean"])

    second_lidar_points = np.array(second_lidar_points, dtype=np.float32)
    master_lidar_points = np.array(master_lidar_points, dtype=np.float32)

    master_to_second_transform = solve_rigid_transform_3d(second_lidar_points, master_lidar_points)

    print("\n---" + "-" * 10 + " Dual Calibration Finished " + "-" * 10 + "---")
    print("Master LiDAR to Camera Transform:")
    print(master_to_camera)
    print("\nMaster LiDAR to Second LiDAR Transform:")
    print(master_to_second_transform)

    return master_to_camera, master_to_second_transform


def solve_rigid_transform_3d(source_points, target_points):
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = target_centroid - R @ source_centroid

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    transformed_points = (R @ source_points.T).T + t
    alignment_error = np.mean(np.linalg.norm(transformed_points - target_points, axis=1))
    print(f"3D-3D alignment RMS error: {alignment_error:.6f}")

    return T


def global_dual_lidar_objective(
    params, master_cam_correspondences, second_cam_correspondences, lidar_lidar_correspondences, K, ocam_model=None
):
    master_rvec = params[:3]
    master_tvec = params[3:6]
    second_rvec = params[6:9]
    second_tvec = params[9:12]

    errors = []

    if master_cam_correspondences:
        master_points_3d = np.array([c[1] for c in master_cam_correspondences], dtype=np.float32)
        master_points_2d = np.array([c[0] for c in master_cam_correspondences], dtype=np.float32)

        if ocam_model is None:
            master_points_proj, _ = cv2.projectPoints(master_points_3d, master_rvec, master_tvec, K, None)
            master_points_proj = master_points_proj.reshape(-1, 2)
        else:
            R_m, _ = cv2.Rodrigues(master_rvec)
            pts_cam_m = (R_m @ master_points_3d.T).T + master_tvec
            master_points_proj = ocam_model.project_points(pts_cam_m)

        master_errors = (master_points_proj - master_points_2d).ravel()
        errors.extend(master_errors)

    if second_cam_correspondences:
        second_points_3d = np.array([c[1] for c in second_cam_correspondences], dtype=np.float32)
        second_points_2d = np.array([c[0] for c in second_cam_correspondences], dtype=np.float32)

        if ocam_model is None:
            second_points_proj, _ = cv2.projectPoints(second_points_3d, second_rvec, second_tvec, K, None)
            second_points_proj = second_points_proj.reshape(-1, 2)
        else:
            R_s, _ = cv2.Rodrigues(second_rvec)
            pts_cam_s = (R_s @ second_points_3d.T).T + second_tvec
            second_points_proj = ocam_model.project_points(pts_cam_s)

        second_errors = (second_points_proj - second_points_2d).ravel()
        errors.extend(second_errors)

    if lidar_lidar_correspondences:
        master_R, _ = cv2.Rodrigues(master_rvec)
        second_R, _ = cv2.Rodrigues(second_rvec)

        master_transform = np.eye(4)
        master_transform[:3, :3] = master_R
        master_transform[:3, 3] = master_tvec

        second_transform = np.eye(4)
        second_transform[:3, :3] = second_R
        second_transform[:3, 3] = second_tvec

        second_to_master = np.linalg.inv(master_transform) @ second_transform

        for second_3d_key, corr_data in lidar_lidar_correspondences.items():
            second_point = np.array(second_3d_key)
            master_point = corr_data["master_3d_mean"]

            second_homogeneous = np.append(second_point, 1.0)
            transformed_point = (second_to_master @ second_homogeneous)[:3]

            alignment_error = transformed_point - master_point
            errors.extend(alignment_error)

    return np.array(errors)


def calibrate_dual_lidar_global(
    master_cam_correspondences,
    second_cam_correspondences,
    lidar_lidar_correspondences,
    K,
    initial_master_transform=None,
    initial_second_transform=None,
    lsq_method="lm",
    lsq_verbose=2,
    ocam_model=None, # 新增参数
):
    print("---" + "-" * 10 + " Starting Global Dual LiDAR Calibration " + "-" * 10 + "---")

    master_count = len(master_cam_correspondences)
    second_count = len(second_cam_correspondences)
    lidar_count = len(lidar_lidar_correspondences)

    if master_count < 4:
        print("Error: Need at least 4 master LiDAR-camera correspondences.")
        return np.eye(4), np.eye(4)

    if second_count < 4 and lidar_count < 3:
        print("Error: Need at least 4 second LiDAR-camera OR 3 LiDAR-LiDAR correspondences.")
        return np.eye(4), np.eye(4)

    print(f"Optimizing with {master_count} master-cam, {second_count} second-cam, {lidar_count} lidar-lidar correspondences")

    if initial_master_transform is not None:
        master_rvec_init, _ = cv2.Rodrigues(initial_master_transform[:3, :3])
        master_tvec_init = initial_master_transform[:3, 3]
    else:
        master_rvec_init = np.zeros(3)
        master_tvec_init = np.zeros(3)

    if initial_second_transform is not None:
        second_rvec_init, _ = cv2.Rodrigues(initial_second_transform[:3, :3])
        second_tvec_init = initial_second_transform[:3, 3]
    else:
        second_rvec_init = np.zeros(3)
        second_tvec_init = np.zeros(3)

    initial_params = np.concatenate(
        [
            master_rvec_init.ravel(),
            master_tvec_init.ravel(),
            second_rvec_init.ravel(),
            second_tvec_init.ravel(),
        ]
    )

    print(f"Initial parameters shape: {initial_params.shape}")
    print("Running global optimization...")

    result = least_squares(
        global_dual_lidar_objective,
        initial_params,
        args=(
            master_cam_correspondences,
            second_cam_correspondences,
            lidar_lidar_correspondences,
            K,
            ocam_model 
        ),
        method=lsq_method,
        verbose=lsq_verbose,
    )

    optimized_params = result.x
    master_rvec_opt = optimized_params[:3]
    master_tvec_opt = optimized_params[3:6]
    second_rvec_opt = optimized_params[6:9]
    second_tvec_opt = optimized_params[9:12]

    master_R_opt, _ = cv2.Rodrigues(master_rvec_opt)
    second_R_opt, _ = cv2.Rodrigues(second_rvec_opt)

    master_transform = np.eye(4)
    master_transform[:3, :3] = master_R_opt
    master_transform[:3, 3] = master_tvec_opt

    second_transform = np.eye(4)
    second_transform[:3, :3] = second_R_opt
    second_transform[:3, 3] = second_tvec_opt

    final_errors = global_dual_lidar_objective(
        optimized_params,
        master_cam_correspondences,
        second_cam_correspondences,
        lidar_lidar_correspondences,
        K,
        ocam_model 
    )
    final_rms_error = np.sqrt(np.mean(final_errors**2))

    print("\n---" + "-" * 10 + " Global Calibration Finished " + "-" * 10 + "---")
    print(f"Final RMS error: {final_rms_error:.6f}")
    print(f"Optimization success: {result.success}")
    print(f"Optimization message: {result.message}")
    print("Master LiDAR to Camera Transform:")
    print(master_transform)
    print("Second LiDAR to Camera Transform:")
    print(second_transform)

    return master_transform, second_transform