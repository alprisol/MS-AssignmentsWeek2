import numpy as np


def S(w):
    """
    Compute the skew-symmetric matrix of a 3-element vector.

    Parameters:
        w (ndarray): A 3-element vector.

    Returns:
        ndarray: A 3x3 skew-symmetric matrix.
    """
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def T(q):
    """
    Compute the 4x4 transformation matrix for quaternion multiplication.

    Parameters:
        q (ndarray): A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3].

    Returns:
        ndarray: A 4x4 transformation matrix.
    """
    eta = q[0]
    epsilon = np.array(q[1:4])
    I = np.eye(3)
    top_left = np.array([[eta]])
    top_right = -epsilon.reshape(1, 3)
    bottom_left = epsilon.reshape(3, 1)
    bottom_right = eta * I + S(epsilon)
    return np.block([[top_left, top_right], [bottom_left, bottom_right]])


def calculate_quaternion_product(q1, q2):
    """
    Compute the quaternion product q1 ⊗ q2.

    Parameters:
        q1 (array-like): The first quaternion [w, x, y, z].
        q2 (array-like): The second quaternion [w, x, y, z].

    Returns:
        ndarray: The resulting quaternion.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z], dtype=float)


def calculate_inverse_quaternion(q):
    """
    Compute the inverse (conjugate) of a quaternion.

    Parameters:
        q (ndarray): A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3].

    Returns:
        ndarray: The inverse quaternion.
    """
    q_inv = q.copy()
    q_inv[1:] = -q[1:]
    return q_inv


def calculate_rotation_matrix_from_quaternion(q):
    """
    Compute the 3x3 rotation matrix corresponding to a quaternion.

    Parameters:
        q (ndarray): A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3].

    Returns:
        ndarray: A 3x3 rotation matrix.
    """
    eta = q[0]
    epsilon = np.array(q[1:4])
    S_epsilon = S(epsilon)
    return np.eye(3) + 2 * eta * S_epsilon + 2 * (S_epsilon @ S_epsilon)


def calculate_quaternion_from_rotation_matrix(R):
    """
    Compute the quaternion from a 3x3 rotation matrix.

    Parameters:
        R (ndarray): A 3x3 rotation matrix.

    Returns:
        ndarray: A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3].
    """
    if not (np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)):
        raise ValueError("Input matrix is not a valid rotation matrix.")

    trace = np.trace(R)
    q = np.zeros(4)

    if trace > 0:
        S_val = 2.0 * np.sqrt(1.0 + trace)
        q[0] = 0.25 * S_val
        q[1] = (R[2, 1] - R[1, 2]) / S_val
        q[2] = (R[0, 2] - R[2, 0]) / S_val
        q[3] = (R[1, 0] - R[0, 1]) / S_val
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S_val = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q[0] = (R[2, 1] - R[1, 2]) / S_val
        q[1] = 0.25 * S_val
        q[2] = (R[0, 1] + R[1, 0]) / S_val
        q[3] = (R[0, 2] + R[2, 0]) / S_val
    elif R[1, 1] > R[2, 2]:
        S_val = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q[0] = (R[0, 2] - R[2, 0]) / S_val
        q[1] = (R[0, 1] + R[1, 0]) / S_val
        q[2] = 0.25 * S_val
        q[3] = (R[1, 2] + R[2, 1]) / S_val
    else:
        S_val = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q[0] = (R[1, 0] - R[0, 1]) / S_val
        q[1] = (R[0, 2] + R[2, 0]) / S_val
        q[2] = (R[1, 2] + R[2, 1]) / S_val
        q[3] = 0.25 * S_val

    return q


def quaternion_kinematics(q_ob, w_ob_b):
    """
    Compute the derivative of a quaternion given an angular velocity.

    Parameters:
        q_ob (ndarray): A 4-element quaternion [eta, epsilon1, epsilon2, epsilon3].
        w_ob_b (ndarray): A 3-element angular velocity vector in the body frame.

    Returns:
        ndarray: The 4-element quaternion derivative.
    """
    w_quat = np.array([0, w_ob_b[0], w_ob_b[1], w_ob_b[2]])
    return 0.5 * T(q_ob) @ w_quat


def attitude_dynamics(J, q_ob, w_ob_b, w_io_i, w_io_i_dot, R_i_o, tau_a_b, tau_p_b):
    """
    Compute the time derivative of the body angular velocity (expressed in the body frame).

    Parameters:
        J (ndarray): Inertia matrix (3x3).
        q_ob (ndarray): Quaternion [eta, eps1, eps2, eps3] (body relative to orbit).
        w_ob_b (ndarray): Angular velocity of the body with respect to orbit, in the body frame.
        w_io_i (ndarray): Angular velocity of orbit with respect to inertial, in the inertial frame.
        w_io_i_dot (ndarray): Time derivative of w_io_i.
        R_i_o (ndarray): Rotation matrix from orbit to inertial frame.
        tau_a_b (ndarray): Actuator torque in the body frame.
        tau_p_b (ndarray): Disturbance torque in the body frame.

    Returns:
        ndarray: Time derivative of the body angular velocity (in the body frame).
    """
    R_b_o = calculate_rotation_matrix_from_quaternion(q_ob)
    R_o_i = R_i_o.T
    R_b_i = R_b_o @ R_o_i
    w_ib_b = w_ob_b + R_b_i @ w_io_i

    term1 = -S(w_ib_b) @ J @ w_ib_b
    term2 = tau_a_b + tau_p_b
    term3 = J @ S(w_ib_b) @ (R_b_i @ w_io_i)
    term4 = J @ (R_b_i @ w_io_i_dot)

    rhs = term1 + term2 + term3 - term4
    return np.linalg.inv(J) @ rhs


def calculate_euler_angles_from_quaternion(q_ab, degrees=True):
    """
    Compute Euler angles (roll, pitch, yaw) from a quaternion.

    Parameters:
        q_ab (array-like): A 4-element quaternion [q0, q1, q2, q3] with q0 as the scalar part.
        degrees (bool, optional): If True, output angles are in degrees.

    Returns:
        ndarray: Euler angles [roll, pitch, yaw].
    """
    q0, q1, q2, q3 = q_ab

    phi = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1**2 + q2**2))
    s = 2.0 * (q0 * q2 - q1 * q3)
    theta = np.arcsin(np.clip(s, -1.0, 1.0))
    psi = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2**2 + q3**2))

    angles = np.array([phi, theta, psi])
    if degrees:
        angles = np.degrees(angles)
    return angles


def calculate_euler_angles_from_rotation_matrix(R, out_in_rad=True):
    """
    Compute Euler angles (roll, pitch, yaw) from a rotation matrix using the x-y-z convention.

    Parameters:
        R (ndarray): A 3x3 rotation matrix.
        out_in_rad (bool, optional): If True, output angles are in radians; otherwise in degrees.

    Returns:
        list: Euler angles [roll, pitch, yaw].
    """
    phi = np.arctan2(R[2, 1], R[2, 2])
    s = -R[2, 0]
    theta = np.arcsin(np.clip(s, -1.0, 1.0))
    psi = np.arctan2(R[1, 0], R[0, 0])
    angles = [phi, theta, psi]
    if not out_in_rad:
        angles = np.degrees(angles)
    return angles
