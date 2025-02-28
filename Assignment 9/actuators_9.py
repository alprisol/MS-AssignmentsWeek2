import numpy as np
from attitude_dynamics_9 import S


def attitude_control_using_thrusters(tau_d_b, max_thrust):
    """
    Maps the desired control torque to thruster firings using a fixed allocation matrix
    and bang-bang modulation with a 0.1 deadzone.

    Parameters
    ----------
    tau_d_b : numpy.ndarray
        Desired control torque vector in the body frame (shape: (3,)).
    max_thrust : float
        Maximum available thrust for each thruster.

    Returns
    -------
    tau_a_b : numpy.ndarray
        Actuation torque computed from the thruster firings (shape: (3,)).
    thruster_firings : numpy.ndarray
        Thruster firing commands after bang-bang modulation (shape: (4,)).
    """
    B = np.array(
        [
            [-np.sqrt(2) / 5, np.sqrt(2) / 5, np.sqrt(2) / 5, -np.sqrt(2) / 5],
            [np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4],
            [-np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4],
        ]
    )
    B_pinv = np.linalg.pinv(B)
    u_d = B_pinv.dot(tau_d_b)
    deadzone = 0.1
    thruster_firings = np.where(u_d > deadzone, max_thrust, 0)
    tau_a_b = B.dot(thruster_firings)
    return tau_a_b, thruster_firings


def attitude_control_using_reaction_wheels_in_tethrahedron(
    tau_d_b, w_ib_b, w_bw_b, J_w, max_RPM
):
    """
    Computes reaction wheel accelerations and the resulting satellite actuation torque
    for a tetrahedron-configured reaction wheel assembly.

    Parameters
    ----------
    tau_d_b : numpy.ndarray
        Desired control torque (3,).
    w_ib_b : numpy.ndarray
        Body angular velocity (3,).
    w_bw_b : numpy.ndarray
        Reaction wheel speeds (4,).
    J_w : float
        Reaction wheel inertia.
    max_RPM : float
        Maximum allowable wheel speed in RPM.

    Returns
    -------
    w_bw_dot : numpy.ndarray
        Reaction wheel angular accelerations (4,).
    tau_a_b : numpy.ndarray
        Satellite actuation torque (3,).
    """
    sqrt3 = np.sqrt(3)
    B = (sqrt3 / 3) * np.array([[1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, 1, 1]])
    G = J_w * B
    G_pinv = np.linalg.pinv(G)
    u = -G_pinv @ tau_d_b
    dynamic_term = -G_pinv @ (S(w_ib_b) @ (G @ w_bw_b))
    w_bw_dot = dynamic_term + u

    max_rad = max_RPM * 2 * np.pi / 60.0
    for i in range(len(w_bw_b)):
        if w_bw_b[i] >= max_rad and w_bw_dot[i] > 0:
            w_bw_dot[i] = 0.0
            u[i] = 0.0
        elif w_bw_b[i] <= -max_rad and w_bw_dot[i] < 0:
            w_bw_dot[i] = 0.0
            u[i] = 0.0

    tau_a_b = -G @ u
    return w_bw_dot, tau_a_b


def attitude_control_using_magnetic_torquers(tau_d_b, b_b, A, N, i_max):
    """
    Computes the actuation torque from magnetic torquers given the desired torque,
    local magnetic field, coil area, number of turns, and maximum current.

    Parameters
    ----------
    tau_d_b : numpy.ndarray
        Desired control torque in the body frame (shape: (3,)).
    b_b : numpy.ndarray
        Earth's magnetic field vector in the body frame (shape: (3,)).
    A : float
        Coil area of the magnetic torquers.
    N : float
        Number of turns in the coils.
    i_max : float
        Maximum allowable current.

    Returns
    -------
    tau_a_b : numpy.ndarray
        Actuation torque computed from the magnetic torquer model (shape: (3,)).
    """
    tau_d_b = np.asarray(tau_d_b).flatten()
    b_b = np.asarray(b_b).flatten()
    norm_b2 = np.dot(b_b, b_b)
    if norm_b2 < 1e-8:
        return np.zeros(3)
    m_b = S(b_b) @ tau_d_b / norm_b2
    currents = m_b / (N * A)
    currents_sat = np.clip(currents, -i_max, i_max)
    m_b_sat = N * A * currents_sat
    tau_b_m = S(m_b_sat) @ b_b
    tau_a_b = np.sign(tau_d_b) * np.abs(tau_b_m)
    return tau_a_b
