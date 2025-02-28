import numpy as np

from orbital_mechanics_9 import calculate_rotation_matrix_from_quaternion


def measure_sensor(v_b, sensor_accuracy):
    """
    Simulates a sensor measurement by rotating the true unit vector v_b by
    a small random angle. The angle noise is drawn from a normal distribution
    with standard deviation = sensor_accuracy/2.

    Parameters:
    -----------
    v_b : ndarray
        The true (unit) vector (e.g., in the body frame).
    sensor_accuracy : float
        Sensor noise parameter (in radians). For example:
          - Magnetic field: 5*np.pi/180
          - Sun sensor: 0.01*np.pi/180

    Returns:
    --------
    v_b_measurement : ndarray
        The measured (noisy) unit vector.
    """
    angle_standard_deviation = sensor_accuracy / 2.0
    # Calculate noise in angle (radians) using a normal distribution:
    angle_noise = np.random.normal(0, angle_standard_deviation)

    # Create a random rotation axis (unit vector):
    random_axis_of_rotation = np.random.randn(3)
    k_noise = random_axis_of_rotation / np.linalg.norm(random_axis_of_rotation)

    # Construct the noise quaternion:
    q_mb = np.array(
        [
            np.cos(angle_noise / 2.0),
            k_noise[0] * np.sin(angle_noise / 2.0),
            k_noise[1] * np.sin(angle_noise / 2.0),
            k_noise[2] * np.sin(angle_noise / 2.0),
        ]
    )

    # Compute the rotation matrix from the noise quaternion:
    R_m_b = calculate_rotation_matrix_from_quaternion(q_mb)

    # Apply the rotation to v_b:
    v_b_measurement = R_m_b @ v_b
    return v_b_measurement


def quest_algorithm(b_b, b_o, s_b, s_o, a1=0.01, a2=0.01):
    """
    Computes the optimal quaternion that aligns the body and orbit frames
    using the QUEST algorithm with two vector measurements.

    Parameters:
    -----------
    b_b : ndarray
        Body-frame measurement vector for the magnetic field.
    b_o : ndarray
        Reference-frame (orbit) vector corresponding to the magnetic field.
    s_b : ndarray
        Body-frame measurement vector for the sun sensor.
    s_o : ndarray
        Reference-frame (orbit) vector corresponding to the sun sensor.
    a1 : float, optional
        Weight (gain) for the magnetic field measurement (default 0.01).
    a2 : float, optional
        Weight (gain) for the sun sensor measurement (default 0.01).

    Returns:
    --------
    q_ob : ndarray
        The estimated unit quaternion representing the rotation from the body
        frame to the orbit frame.
    """
    # Build the attitude profile matrix B:
    B = a1 * np.outer(b_b, b_o) + a2 * np.outer(s_b, s_o)

    # Compute sigma (trace of B) and the symmetric matrix C:
    sigma = np.trace(B)
    C = B + B.T

    # Compute the vector z (from the skew-symmetric parts of B):
    z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])

    # Construct the 4x4 K matrix using np.block():
    # K = [ sigma       z^T ]
    #     [  z      C - sigma*I ]
    K = np.block([[sigma, z.reshape(1, -1)], [z.reshape(-1, 1), C - sigma * np.eye(3)]])

    # Eigen-decomposition of K:
    eigenvalues, eigenvectors = np.linalg.eig(K)
    idx_max = np.argmax(eigenvalues)
    q_ob_hat = eigenvectors[:, idx_max]

    # Normalize the quaternion:
    q_ob = q_ob_hat / np.linalg.norm(q_ob_hat)

    return q_ob


def quaternion_sign_correction(q_new, q_old):
    """
    Corrects the sign of the new quaternion relative to the previous estimate
    to avoid sign jumps.

    Parameters:
    -----------
    q_new : ndarray
        Newly computed quaternion.
    q_old : ndarray
        Previous quaternion estimate.

    Returns:
    --------
    q_new : ndarray
        The sign-corrected quaternion.
    """
    if np.dot(q_old, q_new) < 0.0:
        q_new = -q_new
    return q_new


def match_quaternion_sign_to_reference(q_est, q_ref):
    """
    Adjusts the sign of the estimated quaternion to match that of the reference.

    Parameters:
    -----------
    q_est : ndarray
        The estimated quaternion.
    q_ref : ndarray
        The reference (or true) quaternion.

    Returns:
    --------
    q_est : ndarray
        The estimated quaternion with sign matched to the reference.
    """
    if np.dot(q_est, q_ref) < 0.0:
        q_est = -q_est
    return q_est


# ----------------------------
# Example demonstration:
if __name__ == "__main__":
    # For reproducibility:
    np.random.seed(42)

    # True vectors (from assignment/teacher email)
    s_o = np.array([-0.6087886, 0.79079056, -0.06345657])
    s_b = np.array([0.0, 1.0, 0.0])

    b_o = np.array([0.72802773, 0.52510482, -0.44072731])
    b_b = np.array([1.0, 0.0, 0.0])

    # Sensor noise parameters (in radians)
    magnetic_sensor_accuracy = 5 * np.pi / 180  # 5 degrees noise for magnetic field
    sun_sensor_accuracy = 0.01 * np.pi / 180  # 0.01 degrees noise for sun sensor

    # Simulate sensor measurements using the provided noise levels:
    # b_measured = measure_sensor(b_b, magnetic_sensor_accuracy)
    # s_measured = measure_sensor(s_b, sun_sensor_accuracy)
    b_measured = np.array([0.99928323, -0.03001772, 0.0230643])
    s_measured = np.array([1.59277926e-05, 9.99999997e-01, -7.70648932e-05])

    print("Simulated magnetic field measurement (body frame):")
    print(b_measured)
    print("\nSimulated sun sensor measurement (body frame):")
    print(s_measured)

    # Suppose an initial previous quaternion (e.g., identity rotation)
    q_prev = np.array([1.0, 0.0, 0.0, 0.0])

    # Compute the estimated quaternion using the QUEST algorithm:
    q_est = quest_algorithm(b_measured, b_o, s_measured, s_o, a1=0.01, a2=0.01)

    # Apply sign correction relative to the previous estimate:
    q_est_corrected = quaternion_sign_correction(q_est, q_prev)

    # For demonstration, assume we know a reference (truth) quaternion:
    q_ref = np.array([0.92387953, -0.10227645, 0.2045529, 0.30682935])
    q_est_matched = match_quaternion_sign_to_reference(q_est_corrected, q_ref)

    print("\nEstimated quaternion from QUEST (before sign correction):")
    print(q_est)
    print("\nEstimated quaternion after sign correction (q_est_corrected):")
    print(q_est_corrected)
    print("\nEstimated quaternion after matching sign to reference (q_est_matched):")
    print(q_est_matched)
