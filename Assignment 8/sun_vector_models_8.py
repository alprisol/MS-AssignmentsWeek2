import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R


# Global NumPy print options
np.set_printoptions(precision=3, suppress=True)


def rotation_matrix_and_euler(v1: np.ndarray, v2: np.ndarray):
    """
    Compute the rotation matrix and XYZ Euler angles required to rotate vector v1 to align with v2.

    Parameters:
        v1 (np.ndarray): Initial 3D vector.
        v2 (np.ndarray): Target 3D vector.

    Returns:
        tuple:
            R_matrix (np.ndarray): 3x3 rotation matrix.
            euler_angles (np.ndarray): Euler angles (radians, XYZ convention).
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3), np.zeros(3)

    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    euler_angles = R.from_matrix(R_matrix).as_euler("xyz", degrees=False)
    return R_matrix, euler_angles


def simple_sun_vector_model(t_days, epsilon_deg=23.5, year_days=365.24):
    """
    Compute the Sun vector in the inertial frame for a given time (in days) since vernal equinox.

    Parameters:
        t_days (float or array): Time in days since vernal equinox.
        epsilon_deg (float): Earth's obliquity in degrees (default 23.5).
        year_days (float): Days in a year (default 365.24).

    Returns:
        np.ndarray: Sun vector [x, y, z] in the inertial frame.
    """
    epsilon = np.radians(epsilon_deg)
    omega_i_s = 2.0 * np.pi / year_days  # Angular speed in rad/day
    theta = omega_i_s * t_days
    s_x = np.cos(theta)
    s_y = np.cos(epsilon) * np.sin(theta)
    s_z = np.sin(epsilon) * np.sin(theta)
    return np.array([s_x, s_y, s_z])


def calculate_julian_date(simulation_time, initial_datetime):
    """
    Calculate the Julian Date after simulation_time seconds have elapsed from an initial datetime.

    Parameters:
        simulation_time (float or int): Time in seconds since simulation start.
        initial_datetime (datetime): Start datetime of the simulation.

    Returns:
        float: Julian Date corresponding to the given time.
    """
    current_time = initial_datetime + timedelta(seconds=simulation_time)
    yr, mo, d = current_time.year, current_time.month, current_time.day
    h, min_, s = current_time.hour, current_time.minute, current_time.second
    jd_int_part = (
        367 * yr
        - math.floor(7 * (yr + math.floor((mo + 9) / 12)) / 4)
        + math.floor(275 * mo / 9)
        + d
        + 1721013.5
    )
    day_fraction = (h + min_ / 60.0 + s / 3600.0) / 24.0
    return jd_int_part + day_fraction


def reduce_angle_deg(angle_deg):
    """
    Reduce an angle in degrees to the [0, 360) range.

    Parameters:
        angle_deg (float): Angle in degrees.

    Returns:
        float: Reduced angle in degrees.
    """
    return angle_deg % 360.0


def calculate_advanced_sun_vector_model(simulation_time, initial_datetime):
    """
    Compute the Sun vector (in AU) in an Earth-centered inertial frame using an advanced model.

    Parameters:
        simulation_time (float or int): Time in seconds since simulation start.
        initial_datetime (datetime): Start datetime of the simulation.

    Returns:
        tuple: (x, y, z) position of the Sun in AU.
    """
    JD = calculate_julian_date(simulation_time, initial_datetime)
    T_UT1 = (JD - 2451545.0) / 36525.0  # Julian centuries since J2000.0

    lambda_M_deg = reduce_angle_deg(280.460 + 36000.771 * T_UT1)
    M_deg = reduce_angle_deg(357.5277233 + 35999.05034 * T_UT1)
    lambda_M = math.radians(lambda_M_deg)
    M = math.radians(M_deg)

    lambda_eclip_deg = reduce_angle_deg(
        lambda_M_deg + 1.914666471 * math.sin(M) + 0.019994643 * math.sin(2 * M)
    )
    lambda_eclip = math.radians(lambda_eclip_deg)
    r = 1.000140612 - 0.016708617 * math.cos(M) - 0.000139589 * math.cos(2 * M)
    eps_deg = 23.439291 - 0.0130042 * T_UT1
    eps_rad = math.radians(eps_deg)

    x = r * math.cos(lambda_eclip)
    y = r * math.cos(eps_rad) * math.sin(lambda_eclip)
    z = r * math.sin(eps_rad) * math.sin(lambda_eclip)
    return (x, y, z)


def main():
    # Example 0: Compare sun vectors at key points using the simple model
    t_days_values = [0, 93, 186, 276]
    sun_vectors = [simple_sun_vector_model(d) for d in t_days_values]
    comparisons = [
        (sun_vectors[0], sun_vectors[1], "1st day of Spring vs. Summer Solstice"),
        (sun_vectors[1], sun_vectors[2], "Summer Solstice vs. 1st day of Autumn"),
        (sun_vectors[2], sun_vectors[3], "1st day of Autumn vs. Winter Solstice"),
        (sun_vectors[3], sun_vectors[0], "Winter Solstice vs. 1st day of Spring"),
        (sun_vectors[3], sun_vectors[1], "Winter Solstice vs. Summer Solstice"),
        (sun_vectors[2], sun_vectors[0], "1st day of Autumn vs. 1st day of Spring"),
    ]
    for vec_a, vec_b, msg in comparisons:
        _, euler_angles = rotation_matrix_and_euler(vec_a, vec_b)
        print(f"{msg}: {np.degrees(euler_angles)}")

    print()

    # Example 1: Calculate Julian Date for a given datetime and simulation time
    start_time = datetime(2025, 2, 3, 9, 30, 5)
    simulation_seconds = 0
    jd = calculate_julian_date(simulation_seconds, start_time)
    print(
        f"In date {start_time}, after {simulation_seconds} seconds, Julian Date is {jd}\n"
    )

    # Example 2: Compute advanced sun vector model
    test_datetime = datetime(2019, 1, 8, 10, 43, 15)
    test_seconds = 0
    sx, sy, sz = calculate_advanced_sun_vector_model(test_seconds, test_datetime)
    r_mag = math.sqrt(sx**2 + sy**2 + sz**2)
    sx_norm, sy_norm, sz_norm = sx / r_mag, sy / r_mag, sz / r_mag
    print(f"In date {test_datetime}, after {test_seconds} seconds:")
    print("Sun vector in AU =", (sx, sy, sz))
    print("Normalized sun vector =", (sx_norm, sy_norm, sz_norm))


if __name__ == "__main__":
    main()
