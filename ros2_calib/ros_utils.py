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

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from . import tf_transformations as transformations


# Mock ROS2 message types
@dataclass
class PointField:
    name: str = ""
    offset: int = 0
    datatype: int = 0
    count: int = 0

    INT8 = 1
    UINT8 = 2
    INT16 = 3
    UINT16 = 4
    INT32 = 5
    UINT32 = 6
    FLOAT32 = 7
    FLOAT64 = 8


@dataclass
class Header:
    stamp: int = 0
    frame_id: str = ""


@dataclass
class PointCloud2:
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0
    fields: List[PointField] = field(default_factory=list)
    is_bigendian: bool = False
    point_step: int = 0
    row_step: int = 0
    data: bytes = b""
    is_dense: bool = False


@dataclass
class CameraInfo:
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0
    distortion_model: str = ""
    d: List[float] = field(default_factory=list)
    k: List[float] = field(default_factory=list)
    r: List[float] = field(default_factory=list)
    p: List[float] = field(default_factory=list)


@dataclass
class Image:
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0
    encoding: str = ""
    is_bigendian: bool = False
    step: int = 0
    data: bytes = b""


# --- Copied and adapted from ros2_numpy.point_cloud2 ---

DUMMY_FIELD_PREFIX = "__"

type_mappings = [
    (PointField.INT8, np.dtype("int8")),
    (PointField.UINT8, np.dtype("uint8")),
    (PointField.INT16, np.dtype("int16")),
    (PointField.UINT16, np.dtype("uint16")),
    (PointField.INT32, np.dtype("int32")),
    (PointField.UINT32, np.dtype("uint32")),
    (PointField.FLOAT32, np.dtype("float32")),
    (PointField.FLOAT64, np.dtype("float64")),
]
pftype_to_nptype = dict(type_mappings)


def fields_to_dtype(fields, point_step):
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            np_dtype_list.append(("%s%d" % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_to_nptype[f.datatype].itemsize * f.count

    while offset < point_step:
        np_dtype_list.append(("%s%d" % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list


def pointcloud2_to_array(cloud_msg, squeeze=True):
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)
    cloud_arr = cloud_arr[
        [
            fname
            for fname, _type in dtype_list
            if not (fname[: len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)
        ]
    ]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def pointcloud2_to_structured_array(cloud_msg, remove_nans=True):
    cloud_arr = pointcloud2_to_array(cloud_msg, squeeze=False)
    if remove_nans:
        # Check for x, y, z fields before attempting to filter NaNs
        if (
            "x" in cloud_arr.dtype.names
            and "y" in cloud_arr.dtype.names
            and "z" in cloud_arr.dtype.names
        ):
            mask = (
                np.isfinite(cloud_arr["x"])
                & np.isfinite(cloud_arr["y"])
                & np.isfinite(cloud_arr["z"])
            )
            cloud_arr = cloud_arr[mask]
    return cloud_arr


# --- Copied and adapted from ros2_numpy.image ---

name_to_dtypes = {
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "bgr16": (np.uint16, 3),
    "bgra16": (np.uint16, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "bayer_rggb8": (np.uint8, 1),
    "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1),
    "bayer_grbg8": (np.uint8, 1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),
    "8UC1": (np.uint8, 1),
    "8UC2": (np.uint8, 2),
    "8UC3": (np.uint8, 3),
    "8UC4": (np.uint8, 4),
    "8SC1": (np.int8, 1),
    "8SC2": (np.int8, 2),
    "8SC3": (np.int8, 3),
    "8SC4": (np.int8, 4),
    "16UC1": (np.uint16, 1),
    "16UC2": (np.uint16, 2),
    "16UC3": (np.uint16, 3),
    "16UC4": (np.uint16, 4),
    "16SC1": (np.int16, 1),
    "16SC2": (np.int16, 2),
    "16SC3": (np.int16, 3),
    "16SC4": (np.int16, 4),
    "32SC1": (np.int32, 1),
    "32SC2": (np.int32, 2),
    "32SC3": (np.int32, 3),
    "32SC4": (np.int32, 4),
    "32FC1": (np.float32, 1),
    "32FC2": (np.float32, 2),
    "32FC3": (np.float32, 3),
    "32FC4": (np.float32, 4),
    "64FC1": (np.float64, 1),
    "64FC2": (np.float64, 2),
    "64FC3": (np.float64, 3),
    "64FC4": (np.float64, 4),
}


def image_to_numpy(msg):
    if msg.encoding not in name_to_dtypes:
        raise TypeError("Unrecognized encoding {}".format(msg.encoding))

    dtype_class, channels = name_to_dtypes[msg.encoding]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
    shape = (msg.height, msg.width, channels)

    data = np.frombuffer(msg.data, dtype=dtype).reshape(shape)
    data.strides = (msg.step, dtype.itemsize * channels, dtype.itemsize)

    if channels == 1:
        data = data[..., 0]
    return data


# Mock ROS2 geometry message types
class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Transform:
    def __init__(self, translation=None, rotation=None):
        self.translation = translation if translation is not None else Vector3()
        self.rotation = rotation if rotation is not None else Quaternion()


class TransformStamped:
    def __init__(self, header=None, child_frame_id="", transform=None):
        self.header = header if header is not None else Header()
        self.child_frame_id = child_frame_id
        self.transform = transform if transform is not None else Transform()


class TFMessage:
    def __init__(self, transforms=None):
        self.transforms = transforms if transforms is not None else []


# Transformation conversion utilities (adapted from ros2_numpy)
def numpify(msg):
    """Convert a ROS message to numpy array."""
    if isinstance(msg, Vector3):
        return np.array([msg.x, msg.y, msg.z], dtype=np.float64)
    elif isinstance(msg, Point):
        return np.array([msg.x, msg.y, msg.z], dtype=np.float64)
    elif isinstance(msg, Quaternion):
        return np.array([msg.x, msg.y, msg.z, msg.w], dtype=np.float64)
    else:
        return msg


def vector3_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg.x, msg.y, msg.z, 0])
    else:
        return np.array([msg.x, msg.y, msg.z])


def point_to_numpy(msg, hom=False):
    if hom:
        return np.array([msg.x, msg.y, msg.z, 1])
    else:
        return np.array([msg.x, msg.y, msg.z])


def quat_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z, msg.w])


def transform_to_numpy(transform: Transform) -> np.ndarray:
    """
    Converts a ROS Transform message to a 4x4 NumPy transformation matrix.
    """
    translation = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        dtype=np.float64,
    )

    rotation = Rotation.from_quat(
        [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
    )

    # Create an identity matrix with the correct data type
    transform_matrix = np.identity(4, dtype=np.float64)

    transform_matrix[:3, :3] = rotation.as_matrix()
    transform_matrix[:3, 3] = translation

    return transform_matrix


def numpy_to_transform(arr):
    """Convert 4x4 homogeneous transformation matrix to Transform message."""
    assert arr.shape == (4, 4)

    trans = transformations.translation_from_matrix(arr)
    quat = transformations.quaternion_from_matrix(arr)

    return Transform(
        translation=Vector3(x=trans[0], y=trans[1], z=trans[2]),
        rotation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
    )

from dataclasses import dataclass, field
import numpy as np

@dataclass
class OCamModel:
    pol: np.ndarray = field(default_factory=lambda: np.array([]))
    invpol: np.ndarray = field(default_factory=lambda: np.array([]))
    center: tuple = (0.0, 0.0)
    affine: tuple = (1.0, 0.0, 0.0)
    width: int = 0
    height: int = 0

    def project_points(self, pts_3d):
        """Project 3D points (camera frame) to fisheye image plane using the Scaramuzza OCamCalib model.

        Args:
            pts_3d: (N, 3) array of 3-D points **already in the camera coordinate frame**.

        Returns:
            (N, 2) array of pixel coordinates [u, v].

        Model convention (OCamCalib / Scaramuzza):
            - ``invpol`` is the inverse polynomial mapping angle from optical axis θ → sensor radius ρ
              where θ = arccos(z / ‖p‖) is the angle from the optical axis
              (0 on optical axis, π/2 at 90° off-axis), and ρ is in pixels from the
              distortion centre.
            - ``affine`` = (c, d, e) is the 2×2 stretch / affine matrix [[c, d], [e, 1]].
            - ``center`` = (xc, yc) is the distortion centre in pixels.
        """
        x, y, z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
        
        # 3D norm to calculate normalized z for arccos
        norm_3d = np.sqrt(x*x + y*y + z*z)
        norm_3d_safe = np.where(norm_3d < 1e-9, 1.0, norm_3d)
        z_n = z / norm_3d_safe

        # Angle from optical axis (0 on-axis, pi/2 at 90 deg off-axis)
        theta = np.arccos(np.clip(z_n, -1, 1))

        # Evaluate invpol via np.polyval (descending order)
        if len(self.invpol) == 0:
            rho = np.zeros_like(theta)
        else:
            rho = np.polyval(self.invpol, theta)

        # 2D norm in XY plane for unit direction vector in the (x, y) sensor plane
        xy_norm = np.sqrt(x * x + y * y)
        ON_AXIS = xy_norm < 1e-9
        xy_norm_safe = np.where(ON_AXIS, 1.0, xy_norm)

        # Unit direction vector in the (x, y) sensor plane
        dx = np.where(ON_AXIS, 0.0, x / xy_norm_safe)
        dy = np.where(ON_AXIS, 0.0, y / xy_norm_safe)

        # Undistorted sensor coordinates (relative to distortion centre)
        u = dx * rho
        v = dy * rho

        # Apply affine / stretch matrix and add distortion centre
        c, d, e = self.affine
        xc, yc = self.center

        u_pix = c * u + d * v + xc
        v_pix = e * u + v + yc

        return np.vstack((u_pix, v_pix)).T