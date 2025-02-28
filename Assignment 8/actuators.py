import numpy as np


def attitude_control_using_thrusters(tau_d_b, max_thrust):
    """
    Maps the desired control torque (tau_d_b) to thruster firings using
    the provided allocation matrix and bang-bang modulation with a deadzone of 0.1.

    Parameters
    ----------
    tau_d_b : numpy.ndarray
        Desired control torque vector in the body frame (shape: (3,)).
    max_thrust : float
        The maximum available thrust for each thruster.

    Returns
    -------
    tau_b_a : numpy.ndarray
        Actuation torque computed from the thruster firings (shape: (3,)).
    thruster_firings : numpy.ndarray
        Thruster firing commands after applying bang-bang modulation with a deadzone (shape: (4,)).
    """
    # Define the allocation matrix B as given in Equation (2)
    # B maps individual thruster forces to the resulting torque.
    B = np.array(
        [
            [-np.sqrt(2) / 5, np.sqrt(2) / 5, np.sqrt(2) / 5, -np.sqrt(2) / 5],
            [np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4, -np.sqrt(2) / 4],
            [-np.sqrt(2) / 4, -np.sqrt(2) / 4, np.sqrt(2) / 4, np.sqrt(2) / 4],
        ]
    )

    # Compute the Moore-Penrose pseudoinverse of B.
    B_pinv = np.linalg.pinv(B)

    # Map the desired torque to the continuous thruster firing signals.
    u_d = B_pinv.dot(tau_d_b)

    # Apply bang-bang modulation with a deadzone of 0.1.
    # Thruster firing is set to max_thrust if the continuous signal exceeds 0.1, otherwise 0.
    deadzone = 0.1
    thruster_firings = np.where(u_d > deadzone, max_thrust, 0)

    # Calculate the actuation torque resulting from the applied thruster firings.
    tau_b_a = B.dot(thruster_firings)

    return tau_b_a, thruster_firings
