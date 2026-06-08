# MIT License

# Copyright (c) 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
# Copyright (c) 2026 GRASP Lab, University of Pennsylvania (Added Scaramuzza OCamCalib Fisheye Support)

# Acknowledgement: Portions of the modifications in this file, including 
# the adaptation for dual-fisheye datasets and unstitched frame processing, 
# were developed with the assistance of AI (Google Gemini and ChatGPT).

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import numpy as np
import cv2

class OCamCalibModel:
    def __init__(self, calib_json_path, raw_scale_x=1.0, raw_scale_y=1.0, raw_offset_x=0.0, raw_offset_y=0.0):
        self.calib_json_path = calib_json_path
        self.raw_scale_x = float(raw_scale_x)
        self.raw_scale_y = float(raw_scale_y)
        self.raw_offset_x = float(raw_offset_x)
        self.raw_offset_y = float(raw_offset_y)

        with open(calib_json_path, "r") as f:
            self.data = json.load(f)

        self.ss = np.array(self.data["taylor_coefficient"], dtype=np.float64)
        self.center = np.array(self.data["distortion_center"], dtype=np.float64)
        self.stretch = np.array(self.data["stretch_matrix"], dtype=np.float64)

        self.lut_theta, self.lut_rho = self._build_stable_lut()
        
        # We need width and height for GUI representation
        # It's not standard in ocamcalib json, but usually it can be derived or set.
        # Assuming dual-fisheye crop 1920x1920 by default if not specified
        self.width = self.data.get("width", 1920)
        self.height = self.data.get("height", 1920)
        self.distortion_model = "scaramuzza"
        
        # Provide a dummy K for any code that strictly requires shape (3,3) for display purposes
        self.k = [1.0, 0, self.width/2, 0, 1.0, self.height/2, 0, 0, 1.0]
        self.d = []

    def _build_stable_lut(self):
        rho_samples = np.linspace(0.0, 1500.0, 4000)
        z = np.zeros_like(rho_samples, dtype=np.float64)
        power = np.ones_like(rho_samples, dtype=np.float64)

        for coeff in self.ss:
            z += coeff * power
            power *= rho_samples

        theta_samples = np.arctan2(z, rho_samples)
        return theta_samples, rho_samples

    def project_points(self, points_3d, rvec, tvec):
        """
        Projects 3D points using extrinsic (rvec, tvec) and OCamCalib intrinsics.
        Returns: points_2d (N, 1, 2) shaped similar to cv2.projectPoints
        """
        points_3d = np.array(points_3d, dtype=np.float64).reshape(-1, 3)
        R, _ = cv2.Rodrigues(rvec)
        t = np.array(tvec).reshape(3, 1)

        # Apply extrinsic
        P_cam = (R @ points_3d.T + t).T  # Shape: (N, 3)
        vx = P_cam[:, 0]
        vy = P_cam[:, 1]
        vz = P_cam[:, 2]

        rho_s = np.sqrt(vx * vx + vy * vy)
        rho_s_safe = np.where(rho_s < 1e-12, 1e-12, rho_s)

        theta_query = np.arctan2(vz, rho_s_safe)

        theta_src = self.lut_theta[::-1]
        rho_src = self.lut_rho[::-1]

        # In OCamCalib, we only care about valid angles
        # But we project everything anyway
        valid = (theta_query >= theta_src[0]) & (theta_query <= theta_src[-1])

        rho = np.interp(theta_query, theta_src, rho_src)

        u_un = (vx / rho_s_safe) * rho
        v_un = (vy / rho_s_safe) * rho

        S = self.stretch

        u_corr = S[0, 0] * u_un + S[0, 1] * v_un
        v_corr = S[1, 0] * u_un + S[1, 1] * v_un

        xc, yc = self.center

        map_x_calib = u_corr + xc
        map_y_calib = v_corr + yc

        map_x = (map_x_calib * self.raw_scale_x + self.raw_offset_x).astype(np.float32)
        map_y = (map_y_calib * self.raw_scale_y + self.raw_offset_y).astype(np.float32)

        # Let points behind the camera go to large values or NaN? 
        # OCamCalib can map points behind the camera (theta > pi/2)
        
        # Combine back into shape (N, 1, 2)
        pts_2d = np.stack([map_x, map_y], axis=1)
        return pts_2d.reshape(-1, 1, 2), None

