import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R
from orbital_mechanics_6 import *  # Required for orbital mechanics functions

# Global NumPy print options
np.set_printoptions(precision=3, suppress=True)


def rotation_matrix_and_euler(v1: np.ndarray, v2: np.ndarray):
    """
    Compute the rotation matrix and XYZ Euler angles required to rotate vector v1 to align with v2.
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
    The returned vector is a unit vector.
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
    """
    return angle_deg % 360.0


def calculate_advanced_sun_vector_model(simulation_time, initial_datetime):
    """
    Compute the Sun vector (in AU) in an Earth-centered inertial frame using an advanced model.
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


def exercise1():
    """
    Exercise 1:
    Simulate one satellite orbit and compare the Sun vector expressed in the inertial frame
    and in the orbit (RTN) frame. Note that the sun direction in the inertial frame is nearly constant
    over one orbit, but in the orbit frame it appears to rotate because the frame rotates with the satellite.
    """
    # --- Define orbital parameters for a circular orbit ---
    # Earth gravitational parameter in km^3/s^2
    mu = 398600.4418
    earth_radius = 6378.8
    # Earth's radius plus altitude (in km); e.g., 500 km altitude
    a = 6378.0 + 500.0  # semi-major axis in km
    e = 0.0  # circular
    i_deg = 30.0  # inclination in degrees
    i = np.radians(i_deg)
    RAAN = 0.0  # Right ascension of ascending node (deg)
    arg_perigee = 0.0  # Argument of perigee (deg)
    # For a circular orbit, we can define the position in the perifocal frame as:
    #   r_pf = [a*cos(theta), a*sin(theta), 0]
    # and then rotate to ECI using the standard transformation.
    # With RAAN = 0 and arg_perigee = 0, the ECI position becomes:
    #   r = [a*cos(theta),
    #        a*sin(theta)*cos(i),
    #        a*sin(theta)*sin(i)]
    #
    # Angular rate (mean motion):
    n = np.sqrt(mu / a**3)  # rad/s
    T_orbit = 2 * np.pi / n  # orbital period in seconds

    # Create time array over one orbit
    num_points = 200
    t_array = np.linspace(0, T_orbit, num_points)  # seconds
    t_days_array = t_array / 86400.0  # convert seconds to days for the sun model

    # Preallocate arrays for storing sun vector components in both frames
    s_inertial_all = np.zeros((num_points, 3))
    s_orbit_all = np.zeros((num_points, 3))

    # Loop over time steps
    for idx, (t, t_day) in enumerate(zip(t_array, t_days_array)):
        # --- Inertial sun vector (using the simple model) ---
        s_inertial = simple_sun_vector_model(t_day)
        s_inertial_all[idx, :] = s_inertial

        # --- Satellite position and velocity in inertial frame ---
        # True anomaly (for a circular orbit, theta = n*t)
        theta = n * t
        # Position vector in km (using standard formulas for a circular orbit)
        r = np.array(
            [
                a * np.cos(theta),
                a * np.sin(theta) * np.cos(i),
                a * np.sin(theta) * np.sin(i),
            ]
        )
        # Velocity vector in km/s; note that for circular orbits v = sqrt(mu/a)
        v = np.array(
            [-np.sin(theta), np.cos(theta) * np.cos(i), np.cos(theta) * np.sin(i)]
        ) * np.sqrt(mu / a)

        # --- Define the orbit frame (RTN: Radial, Transverse, Normal) ---
        r_hat = r / np.linalg.norm(r)
        h = np.cross(r, v)
        h_hat = h / np.linalg.norm(h)
        # Transverse direction: along-track = h_hat x r_hat
        t_hat = np.cross(h_hat, r_hat)
        # Transformation from inertial to orbit frame: rows are [r_hat, t_hat, h_hat]
        T_inertial_to_orbit = np.vstack((r_hat, t_hat, h_hat))
        # Express the inertial sun vector in the orbit frame:
        s_orbit = T_inertial_to_orbit @ s_inertial
        s_orbit_all[idx, :] = s_orbit

    # --- Plotting ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    time_minutes = t_array / 60.0  # convert seconds to minutes

    # Plot inertial sun vector components (they are nearly constant over one orbit)
    axs[0].plot(time_minutes, s_inertial_all[:, 0], label="x")
    axs[0].plot(time_minutes, s_inertial_all[:, 1], label="y")
    axs[0].plot(time_minutes, s_inertial_all[:, 2], label="z")
    axs[0].set_ylabel("Sun vector components (Inertial)")
    axs[0].set_title("Sun Vector in Inertial Frame (Nearly Constant over One Orbit)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot orbit frame sun vector components (they vary because the orbit frame rotates)
    axs[1].plot(time_minutes, s_orbit_all[:, 0], label="Radial")
    axs[1].plot(time_minutes, s_orbit_all[:, 1], label="Transverse")
    axs[1].plot(time_minutes, s_orbit_all[:, 2], label="Normal")
    axs[1].set_ylabel("Sun vector components (Orbit Frame)")
    axs[1].set_xlabel("Time (minutes)")
    axs[1].set_title("Sun Vector in Orbit (RTN) Frame (Time-Varying)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("Assignment 6/sun_vector_inertial_vs_orbit.png")
    plt.show()

    print("Exercise 1 Explanation:")
    print(
        "In the inertial frame (fixed relative to distant stars) the sun vector changes very little over one orbit"
    )
    print(
        "because the sun’s apparent motion is about 1° per day. However, the orbit frame rotates with the satellite,"
    )
    print(
        "so the same inertial sun direction appears to change (rotate) in the orbit frame."
    )


def exercise2():
    """
    Exercise 2:
    Run a simulation over one year and plot the Sun vector in the inertial frame using an advanced sun vector model.
    """
    # Choose a start date for the simulation.
    start_datetime = datetime(2020, 1, 1, 0, 0, 0)
    # Define simulation time span: one year (in seconds)
    days_in_year = 365.24
    T_year = days_in_year * 86400.0  # seconds in one year

    num_points = 500
    sim_time_array = np.linspace(0, T_year, num_points)  # seconds

    sun_positions = np.zeros((num_points, 3))  # will hold (x, y, z) in AU

    for idx, sim_time in enumerate(sim_time_array):
        sun_positions[idx, :] = np.array(
            calculate_advanced_sun_vector_model(sim_time, start_datetime)
        )

    # --- Plot the sun vector components vs. time ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    time_days = sim_time_array / 86400.0
    ax1.plot(time_days, sun_positions[:, 0], label="x [AU]")
    ax1.plot(time_days, sun_positions[:, 1], label="y [AU]")
    ax1.plot(time_days, sun_positions[:, 2], label="z [AU]")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Sun vector components (AU)")
    ax1.set_title("Advanced Sun Vector Model in Inertial Frame over One Year")
    ax1.legend()
    ax1.grid(True)

    fig1.savefig("Assignment 6/sun_vector_inertial_one_year.png")

    # --- Plot the sun vector trajectory in the x-y plane ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.plot(sun_positions[:, 0], sun_positions[:, 1], "b-")
    ax2.set_xlabel("x [AU]")
    ax2.set_ylabel("y [AU]")
    ax2.set_title("Sun Vector Trajectory in the Inertial (x-y) Plane over One Year")
    ax2.grid(True)
    ax2.axis("equal")

    plt.show()


def main():
    # (The original examples are still here if desired)
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

    # --- Run Exercise 1 and Exercise 2 ---
    exercise1()
    exercise2()


if __name__ == "__main__":
    main()
