"""Pose conversion utilities for robotic action representations.

Provides relative-to-absolute action conversion for the action formats: XVLA, OpenPI and UMI.
The XVLA format uses 20-dimensional action vectors:
  [0:3]   right xyz
  [3:9]   right rot6d
  [9]     right gripper
  [10:13] left xyz
  [13:19] left rot6d
  [19]    left gripper

The OpenPI format uses 20-dimensional action vectors:
  [0:3]   right xyz
  [3:9]   right rot6d
  [9:13]  left xyz
  [13:18] left rot6d
  [18]    right gripper
  [19]    left gripper

The UMI format uses 10-dimensional action vectors:
  [0:3]   xyz
  [3:6]   rot6d
  [6]     gripper
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# ------------------------------------------------------------------
# Rotation helpers
# ------------------------------------------------------------------

def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Args:
        rot6d: shape (6,) or (N, 6).

    Returns:
        Rotation matrix of shape (3, 3) or (N, 3, 3).
    """
    squeeze = rot6d.ndim == 1
    if squeeze:
        rot6d = rot6d[np.newaxis]

    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)
    b2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + 1e-6)
    b3 = np.cross(b1, b2, axis=-1)

    mat = np.stack([b1, b2, b3], axis=-2)  # (N, 3, 3)
    return mat[0] if squeeze else mat


def rotation_matrix_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D rotation representation.

    Args:
        rotation_matrix: shape (3, 3) or (N, 3, 3).

    Returns:
        rot6d of shape (6,) or (N, 6).
    """
    squeeze = rotation_matrix.ndim == 2
    if squeeze:
        rotation_matrix = rotation_matrix[np.newaxis]

    batch_shape = rotation_matrix.shape[:-2]
    rot6d = rotation_matrix[..., :2, :].copy().reshape(batch_shape + (6,))

    return rot6d.squeeze(0) if squeeze else rot6d




# ------------------------------------------------------------------
# XVLA action format indices
# ------------------------------------------------------------------

XVLA_RIGHT_XYZ = [0, 1, 2]
XVLA_RIGHT_ROT6D = [3, 4, 5, 6, 7, 8]
XVLA_RIGHT_GRIPPER = 9
XVLA_LEFT_XYZ = [10, 11, 12]
XVLA_LEFT_ROT6D = [13, 14, 15, 16, 17, 18]
XVLA_LEFT_GRIPPER = 19


# ------------------------------------------------------------------
# OpenPi action format indices
# ------------------------------------------------------------------

OPENPI_RIGHT_XYZ = [0, 1, 2]
OPENPI_RIGHT_ROT6D = [3, 4, 5, 6, 7, 8]
OPENPI_RIGHT_GRIPPER = 18
OPENPI_LEFT_XYZ = [9, 10, 11]
OPENPI_LEFT_ROT6D = [12, 13, 14, 15, 16, 17]
OPENPI_LEFT_GRIPPER = 19

# ------------------------------------------------------------------
# Conversion functions
# ------------------------------------------------------------------

def xvla_relative_to_absolute(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Convert a single XVLA relative action to absolute pose.

    Args:
        state:  (N, 20) robot state (read-only).
        action: (N, 20) relative action (**modified in-place** and returned).

    Returns:
        action with absolute pose, shape (N, 20).
    """
    action[..., XVLA_RIGHT_XYZ] += state[..., XVLA_RIGHT_XYZ]
    action[..., XVLA_LEFT_XYZ] += state[..., XVLA_LEFT_XYZ]

    for rot_idx in (XVLA_RIGHT_ROT6D, XVLA_LEFT_ROT6D):
        act_rot = rot6d_to_rotation_matrix(action[..., rot_idx])
        st_rot = rot6d_to_rotation_matrix(state[..., rot_idx])
        abs_rot = rotation_matrix_to_rot6d(act_rot @ st_rot)
        action[..., rot_idx] = abs_rot

    return action

def openpi_relative_to_absolute(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Convert a single OpenPI relative action to absolute pose.

    Args:
        state:  (N, 20) robot state (read-only).
        action: (N, 20) relative action (**modified in-place** and returned).

    Returns:
        action with absolute pose, shape (N, 20).
    """
    action[..., OPENPI_RIGHT_XYZ] += state[..., OPENPI_RIGHT_XYZ]
    action[..., OPENPI_LEFT_XYZ] += state[..., OPENPI_LEFT_XYZ]

    for rot_idx in (OPENPI_RIGHT_ROT6D, OPENPI_LEFT_ROT6D):
        act_rot = rot6d_to_rotation_matrix(action[..., rot_idx])
        st_rot = rot6d_to_rotation_matrix(state[..., rot_idx])
        abs_rot = rotation_matrix_to_rot6d(act_rot @ st_rot)
        action[..., rot_idx] = abs_rot

    return action

def umi_relative_to_absolute(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Convert a single UMI relative action to absolute pose.

    Args:
        state:  (N, 7) robot state (read-only). [x, y, z, rx, ry, rz, gripper]
        action: (N, 10) relative action (**modified in-place** and returned). [x, y, z, r1, r2, r3, r4, r5, r6, gripper]

    Returns:
        action with absolute pose, shape (N, 10).
    """
    def pos_rot_to_mat(pos, rot):
        shape = pos.shape[:-1]
        mat = np.zeros(shape + (4,4), dtype=pos.dtype)
        mat[...,:3,3] = pos
        mat[...,:3,:3] = rot.as_matrix()
        mat[...,3,3] = 1
        return mat

    def pose10d_to_mat(d10):
        pos = d10[...,:3]
        d6 = d10[...,3:]
        rotmat = rot6d_to_mat(d6)
        out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
        out[...,:3,:3] = rotmat
        out[...,:3,3] = pos
        out[...,3,3] = 1
        return out

    def normalize(vec, eps=1e-12):
        norm = np.linalg.norm(vec, axis=-1)
        norm = np.maximum(norm, eps)
        out = (vec.T / norm).T
        return out
        
    def rot6d_to_mat(d6):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = normalize(a1)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = normalize(b2)
        b3 = np.cross(b1, b2, axis=-1)
        out = np.stack((b1, b2, b3), axis=-2)
        return out

    def mat_to_pose10d(mat):
        pos = mat[...,:3,3]
        rotmat = mat[...,:3,:3]
        d6 = mat_to_rot6d(rotmat)
        d10 = np.concatenate([pos, d6], axis=-1)
        return d10

    def mat_to_rot6d(mat):
        batch_dim = mat.shape[:-2]
        out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
        return out

    state_mat = pos_rot_to_mat(state[..., :3], 
                              R.from_rotvec(state[..., 3:6]))
    action_mat = pose10d_to_mat(action[..., :-1])
    out_mat = np.linalg.inv(state_mat) @ action_mat
    out = np.concatenate([mat_to_pose10d(out_mat), action[..., -1:]], axis=-1)
    return out


# ------------------------------------------------------------------
# High-level batch conversion
# ------------------------------------------------------------------

def convert_to_absolute(
    predictions: np.ndarray,
    targets: np.ndarray,
    states: np.ndarray,
    action_type: str,
) -> tuple:
    """Convert predictions and targets from relative to absolute pose.

    Args:
        predictions: [T, H, D] relative predicted action chunks.
        targets:     [T, H, D] relative target action chunks.
        states:      [T, D] robot state at each timestep.
        action_type: conversion key (``"xvla_relative"``, ``"openpi_relative"`` or ``"umi_relative"``).

    Returns:
        (abs_predictions, abs_targets) each of shape [T, H, D].
    """
    converters = {
        "xvla_relative": xvla_relative_to_absolute,
        "openpi_relative": openpi_relative_to_absolute,
        "umi_relative": umi_relative_to_absolute,
    }

    if action_type not in converters:
        raise ValueError(
            f"Unknown action_type '{action_type}'. "
            f"Supported: {list(converters.keys())}"
        )

    convert_fn = converters[action_type]
    T, H, D = predictions.shape
    _, Ds = states.shape

    states_expanded = np.tile(states[:, np.newaxis, :], (1, H, 1)).reshape(T * H, Ds)

    abs_pred = convert_fn(
        states_expanded, predictions.reshape(T * H, D).copy()
    ).reshape(T, H, D)

    abs_tgt = convert_fn(
        states_expanded, targets.reshape(T * H, D).copy()
    ).reshape(T, H, D)

    return abs_pred, abs_tgt
