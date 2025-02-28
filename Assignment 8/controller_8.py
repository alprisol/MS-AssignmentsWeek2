import numpy as np
from attitude_dynamics_8 import (
    calculate_inverse_quaternion,
    T,
    calculate_rotation_matrix_from_quaternion,
)


def pd_attitude_controller(q_ob, w_ob_b, q_od, w_od_d, kp, kd):
    """
    PD Attitude Controller.

    Computes the commanded torque in the body frame based on the quaternion and angular
    velocity errors between the current and desired attitudes.

    Parameters:
        q_ob   (ndarray, shape (4,)): Current quaternion [eta, eps1, eps2, eps3] (body w.r.t. orbit).
        w_ob_b (ndarray, shape (3,)): Current angular velocity of the body (expressed in the body frame).
        q_od   (ndarray, shape (4,)): Desired quaternion (body w.r.t. orbit).
        w_od_d (ndarray, shape (3,)): Desired angular velocity.
        kp     (float): Proportional gain.
        kd     (float): Derivative gain.

    Returns:
        ndarray, shape (3,): Commanded torque in the body frame.
    """
    # Compute error quaternion (from desired to body) using the inverse of the desired quaternion.
    q_do = calculate_inverse_quaternion(q_od)
    q_db = T(q_do) @ q_ob

    # Extract the vector part of the error quaternion.
    eps_db = q_db[1:]

    # Compute the body-rate error by rotating the desired angular velocity into the body frame.
    R_db = calculate_rotation_matrix_from_quaternion(q_db)
    w_db_b = w_ob_b - (R_db @ w_od_d)

    # Apply the PD control law.
    t_db = -kp * eps_db - kd * w_db_b

    return t_db
