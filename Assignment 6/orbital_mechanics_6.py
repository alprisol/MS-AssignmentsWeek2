import numpy as np
from datetime import datetime
import ppigrf
from attitude_dynamics_6 import (
    calculate_quaternion_product,
    calculate_rotation_matrix_from_quaternion,
)

np.set_printoptions(precision=3, suppress=True)

# Earth constants
a_e = 6378.137  # Semi-major axis (km)
b_e = 6356.725  # Semi-minor axis (km)
omega_ie = 7.292115e-5  # Earth's angular speed (rad/s)
e_e = 0.0818  # Eccentricity of the reference ellipsoid


# --- Orbital Geometry Functions ---
def calculate_eccentricity(ra, rp):
    """
    Calculate the orbital eccentricity.
    """
    return (ra - rp) / (ra + rp)


def calculate_semimajor_axis(ra, rp):
    """
    Calculate the semi-major axis.
    """
    return (ra + rp) / 2


def calculate_mean_motion(a, mu):
    """
    Calculate the mean motion.
    """
    return np.sqrt(mu / a**3)


def calculate_orbital_period(n):
    """
    Calculate the orbital period.
    """
    return 2 * np.pi / n


# --- Kepler's Equation ---
def calculate_eccentric_anomaly(M, e, tolerance=1e-6):
    """
    Calculate the eccentric anomaly using an iterative method.
    """
    E = M  # initial guess
    while True:
        E_new = M + e * np.sin(E)
        if abs(E_new - E) < tolerance:
            break
        E = E_new
    return E


def calculate_true_anomaly(e, E):
    """
    Calculate the true anomaly.
    """
    return np.arccos((np.cos(E) - e) / (1 - e * np.cos(E)))


def calculate_true_anomaly_derivative(e, theta, n):
    """
    Calculate the derivative of the true anomaly.
    """
    return n * (1 + e * np.cos(theta)) ** 2 / (1 - e**2) ** 1.5


# --- Rotation Matrices (Inertial and PQW) ---
def calculate_rotation_matrix_from_inertial_to_pqw(omega, OMEGA, i):
    """
    Calculate the rotation matrix from inertial to PQW frame.
    """
    cos_OMEGA, sin_OMEGA = np.cos(OMEGA), np.sin(OMEGA)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    R_pqw_i = np.array(
        [
            [
                cos_omega * cos_OMEGA - sin_omega * cos_i * sin_OMEGA,
                cos_omega * sin_OMEGA + sin_omega * cos_i * cos_OMEGA,
                sin_omega * sin_i,
            ],
            [
                -sin_omega * cos_OMEGA - cos_omega * cos_i * sin_OMEGA,
                -sin_omega * sin_OMEGA + cos_omega * cos_i * cos_OMEGA,
                cos_omega * sin_i,
            ],
            [sin_i * sin_OMEGA, -sin_i * cos_OMEGA, cos_i],
        ]
    )
    return R_pqw_i


def calculate_rotation_matrix_from_inertial_to_orbit(omega, theta, OMEGA, i):
    """
    Calculate the rotation matrix from inertial to orbit frame.
    """
    return calculate_rotation_matrix_from_inertial_to_pqw(omega + theta, OMEGA, i)


def calculate_rotation_matrix_from_orbit_to_inertial(omega, theta, OMEGA, i):
    """
    Calculate the rotation matrix from orbit frame to inertial frame.
    """
    return calculate_rotation_matrix_from_inertial_to_orbit(omega, theta, OMEGA, i).T


# --- Angular Velocity and Acceleration ---
def calculate_angular_velocity_of_orbit_relative_to_inertial_referenced_in_inertial(
    r, v
):
    """
    Calculate angular velocity of the orbit relative to the inertial frame.
    """
    return np.cross(r, v) / np.dot(r, r)


def calculate_angular_acceleration_of_orbit_relative_to_inertial_referenced_in_inertial(
    r, v, a
):
    """
    Calculate angular acceleration of the orbit relative to the inertial frame.
    """
    return (np.cross(r, a) * np.dot(r, r) - 2 * np.cross(r, v) * np.dot(v, r)) / np.dot(
        r, r
    ) ** 2


# --- PQW Frame Vectors ---
def calculate_radius_vector_in_pqw(a, e, E):
    """
    Calculate the radius vector in the PQW frame.
    """
    return np.array([a * np.cos(E) - a * e, a * np.sqrt(1 - e**2) * np.sin(E), 0])


def calculate_velocity_vector_in_pqw(a, e, n, r, E):
    """
    Calculate the velocity vector in the PQW frame.
    """
    return np.array(
        [-(a**2) * n / r * np.sin(E), a**2 * n / r * np.sqrt(1 - e**2) * np.cos(E), 0]
    )


def calculate_acceleration_vector_in_pqw(a, e, n, r, E):
    """
    Calculate the acceleration vector in the PQW frame.
    """
    return np.array(
        [
            -(a**3) * n**2 / r**2 * np.cos(E),
            -(a**3) * n**2 / r**2 * np.sqrt(1 - e**2) * np.sin(E),
            0,
        ]
    )


# --- Transform Vectors to Inertial Frame ---
def calculate_radius_vector_in_inertial(r_pqw, R_pqw_i):
    """
    Calculate the radius vector in the inertial frame.
    """
    return R_pqw_i.T @ r_pqw


def calculate_velocity_vector_in_inertial(v_pqw, R_pqw_i):
    """
    Calculate the velocity vector in the inertial frame.
    """
    return R_pqw_i.T @ v_pqw


def calculate_acceleration_vector_in_inertial(a_pqw, R_pqw_i):
    """
    Calculate the acceleration vector in the inertial frame.
    """
    return R_pqw_i.T @ a_pqw


# --- Orbital Quaternions ---
def calculate_quaternion_from_orbital_parameters(omega, OMEGA, i, theta, degrees=False):
    """
    Calculate the orbit-to-inertial quaternion from orbital parameters.
    """
    if degrees:
        omega = np.radians(omega)
        OMEGA = np.radians(OMEGA)
        i = np.radians(i)
        theta = np.radians(theta)
    q_OMEGA = np.array([np.cos(OMEGA / 2.0), 0.0, 0.0, np.sin(OMEGA / 2.0)])
    q_i = np.array([np.cos(i / 2.0), np.sin(i / 2.0), 0.0, 0.0])
    q_omega_plus_theta = np.array(
        [np.cos((omega + theta) / 2.0), 0.0, 0.0, np.sin((omega + theta) / 2.0)]
    )
    return calculate_quaternion_product(
        q_OMEGA, calculate_quaternion_product(q_i, q_omega_plus_theta)
    )


# --- Earth and ECEF Conversions ---
def calculate_rotation_matrix_from_inertial_to_ecef(t):
    """
    Calculate the rotation matrix from inertial to ECEF frame.
    """
    theta = omega_ie * t
    R_i_e = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return R_i_e.T


def calculate_lla_from_ecef(r_e, out_in_degrees=True):
    """
    Calculate latitude, longitude, and altitude from ECEF coordinates.
    """
    x, y, z = r_e
    p = np.sqrt(x**2 + y**2)
    mu_angle = np.arctan((z / p) * (1 - e_e**2) ** -1)
    mu_old = 10
    while abs(mu_angle - mu_old) > 1e-6:
        mu_old = mu_angle
        N = a_e**2 / np.sqrt(
            a_e**2 * np.cos(mu_angle) ** 2 + b_e**2 * np.sin(mu_angle) ** 2
        )
        h = p / np.cos(mu_angle) - N
        mu_angle = np.arctan((z / p) * (1 - e_e**2 * (N / (N + h))) ** -1)
    l = np.atan2(y, x)
    latitude = np.degrees(mu_angle) if out_in_degrees else mu_angle
    longitude = np.degrees(l) if out_in_degrees else l
    return latitude, longitude, h


def calculate_rotation_matrix_from_ecef_to_ned(lat, lon, degrees=True):
    """
    Calculate the rotation matrix from ECEF to NED coordinates.
    """
    if degrees:
        lat = np.radians(lat)
        lon = np.radians(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    R_ECEF_NED = np.array(
        [
            [-cos_lon * sin_lat, -sin_lon, -cos_lon * cos_lat],
            [-sin_lon * sin_lat, cos_lon, -sin_lon * cos_lat],
            [cos_lat, 0, -sin_lat],
        ]
    )
    return R_ECEF_NED.T


def calculate_rotation_matrix_from_ned_to_enu():
    """
    Calculate the rotation matrix from NED to ENU coordinates.
    """
    return np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])


# --- Main Execution ---
if __name__ == "__main__":
    # Section 1: Rotation Matrices and Quaternion from Orbital Parameters
    omega = 20 * np.pi / 180
    OMEGA = 10 * np.pi / 180
    theta = 75 * np.pi / 180
    i = 56 * np.pi / 180

    R_o_i = calculate_rotation_matrix_from_inertial_to_orbit(omega, theta, OMEGA, i)
    q_io = calculate_quaternion_from_orbital_parameters(omega, OMEGA, i, theta)
    R_i_o_QUAT = calculate_rotation_matrix_from_quaternion(q_io)
    R_o_i_QUAT = R_i_o_QUAT.T

    print(f"Rotation Matrix from inertial to orbit (Standard):\n{R_o_i}\n")
    print(f"Quaternion from orbital parameters:\n{q_io}\n")
    print(f"Rotation Matrix from inertial to orbit (from Quaternion):\n{R_o_i_QUAT}\n")
    print("\n------------------------------------------------------------------\n")

    # Section 2: Coordinate Transformations and Magnetic Field Computation
    OMEGA = np.radians(0)
    omega = np.radians(0)
    i = np.radians(75)
    theta = np.radians(30)
    t = 30  # seconds

    r_o = np.array([6420.652, 5236.678, 1111.957])
    print(f"Position in orbital frame (km): {r_o}\n")

    R_i_o = calculate_rotation_matrix_from_orbit_to_inertial(omega, theta, OMEGA, i)
    print(f"Rotation Matrix from orbit to inertial:\n{R_i_o}\n")

    r_i = R_i_o @ r_o
    print(f"Position in ECI (km): {r_i}\n")

    R_ECEF_i = calculate_rotation_matrix_from_inertial_to_ecef(t)
    print(f"Rotation Matrix from inertial to ECEF:\n{R_ECEF_i}\n")

    r_ECEF = R_ECEF_i @ r_i
    print(f"Position in ECEF (km): {r_ECEF}\n")

    lat, lon, alt = calculate_lla_from_ecef(r_ECEF)
    print(
        f"Latitude: {round(lat,4)}º\nLongitude: {round(lon,4)}º\nAltitude: {round(alt,4)} km\n"
    )

    R_NED_ECEF = calculate_rotation_matrix_from_ecef_to_ned(lat, lon)
    print(f"Rotation Matrix from ECEF to NED:\n{R_NED_ECEF}\n")
    r_NED = R_NED_ECEF @ r_ECEF
    print(f"Position in NED: {r_NED}\n")

    R_ENU_NED = calculate_rotation_matrix_from_ned_to_enu()
    print(f"Rotation Matrix from NED to ENU:\n{R_ENU_NED}\n")
    r_ENU = R_ENU_NED @ r_NED
    print(f"Position in ENU (km): {r_ENU}\n")

    date = datetime(2025, 1, 10)
    Be, Bn, Bu = ppigrf.igrf(lon, lat, alt, date)
    B_ENU = np.array([Be, Bn, Bu]) * 1e-9

    np.set_printoptions(precision=3, suppress=False)
    B_NED = R_ENU_NED.T @ B_ENU
    B_ECEF = R_NED_ECEF.T @ B_NED
    B_i = R_ECEF_i.T @ B_ECEF
    B_o = R_i_o.T @ B_i

    print(f"B in ENU frame (nT):\n{B_ENU*1e9}\n")
    print(f"B in orbit frame (nT):\n{B_o*1e9}\n")
