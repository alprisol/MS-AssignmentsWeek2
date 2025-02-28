import numpy as np
from datetime import datetime
import ppigrf
from orbital_mechanics_7 import (
    calculate_lla_from_ecef,
    calculate_rotation_matrix_from_orbit_to_inertial,
    calculate_rotation_matrix_from_inertial_to_ecef,
    calculate_rotation_matrix_from_ecef_to_ned,
    calculate_rotation_matrix_from_ned_to_enu,
)

np.set_printoptions(precision=3, suppress=False)


def calculate_magnetic_field_in_orbit_frame(r_i, date, omega, theta, OMEGA, i, t):
    """
    Calculate the magnetic field in the orbit frame.

    This function converts a given position vector (in the inertial frame)
    to latitude, longitude, and altitude, computes the magnetic field in the ENU frame
    using the IGRF model, and then transforms it into the orbit frame.

    Parameters:
        r_i (array-like): Inertial position vector.
        date (datetime): Date for the IGRF model.
        omega (float): Argument of perigee.
        theta (float): True anomaly.
        OMEGA (float): Right ascension of the ascending node.
        i (float): Inclination in radians.
        t (float): Time in seconds (used for ECEF conversion).

    Returns:
        tuple:
            B_ENU (ndarray): Magnetic field vector in the ENU frame (in Teslas).
            B_o (ndarray): Magnetic field vector in the orbit frame (in Teslas).
    """
    # Convert inertial position to LLA (latitude [º], longitude [º], altitude [km])
    latitude, longitude, altitude = calculate_lla_from_ecef(r_i)

    # Obtain magnetic field components in ENU using the IGRF model (output in nT)
    Be, Bn, Bu = ppigrf.igrf(longitude, latitude, altitude, date)
    B_ENU = np.array([Be, Bn, Bu]) * 1e-9  # Convert nT to Teslas

    # Compute rotation matrices for frame transformations
    R_i_o = calculate_rotation_matrix_from_orbit_to_inertial(omega, theta, OMEGA, i)
    R_ECEF_i = calculate_rotation_matrix_from_inertial_to_ecef(t)
    R_NED_ECEF = calculate_rotation_matrix_from_ecef_to_ned(latitude, longitude)
    R_ENU_NED = calculate_rotation_matrix_from_ned_to_enu()

    # Transform the magnetic field from ENU to orbit frame
    B_NED = R_ENU_NED.T @ B_ENU
    B_ECEF = R_NED_ECEF.T @ B_NED
    B_i = R_ECEF_i.T @ B_ECEF
    B_o = R_i_o.T @ B_i

    return B_ENU, B_o


if __name__ == "__main__":
    OMEGA = 0
    omega = 0
    i = np.radians(75)
    theta = np.radians(30)
    t = 30

    r_i = [2938.363, 942.355, 7769.299]
    date = datetime(2025, 1, 10)

    B_ENU, B_o = calculate_magnetic_field_in_orbit_frame(
        r_i, date, omega, theta, OMEGA, i, t
    )
    print(f"B in ENU frame (nT):\n{B_ENU * 1e9}\n")
    print(f"B in orbit frame (nT):\n{B_o * 1e9}\n")
