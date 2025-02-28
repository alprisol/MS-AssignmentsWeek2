import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Import your simulation and visualization functions
from visualization_6 import visualize_scene, animate_satellite
from orbital_mechanics_6 import (
    calculate_semimajor_axis,
    calculate_eccentricity,
    calculate_mean_motion,
    calculate_true_anomaly_derivative,
    calculate_eccentric_anomaly,
    calculate_rotation_matrix_from_inertial_to_pqw,
    calculate_radius_vector_in_pqw,
    calculate_radius_vector_in_inertial,
    calculate_velocity_vector_in_pqw,
    calculate_acceleration_vector_in_pqw,
    calculate_velocity_vector_in_inertial,
    calculate_acceleration_vector_in_inertial,
    calculate_angular_velocity_of_orbit_relative_to_inertial_referenced_in_inertial,
    calculate_angular_acceleration_of_orbit_relative_to_inertial_referenced_in_inertial,
    calculate_rotation_matrix_from_orbit_to_inertial,
    calculate_quaternion_from_orbital_parameters,
)
from attitude_dynamics_6 import (
    quaternion_kinematics,
    attitude_dynamics,
    calculate_euler_angles_from_quaternion,
    calculate_rotation_matrix_from_quaternion,
    calculate_quaternion_product,
    calculate_inverse_quaternion,
)
from controller_6 import pd_attitude_controller
from magnetic_field_6 import calculate_magnetic_field_in_orbit_frame
from sun_vector_models import (
    calculate_advanced_sun_vector_model,
    rotation_matrix_and_euler,
)

np.set_printoptions(precision=3, suppress=True)

directory_path = "Assignment 6/"


# ------------------------------------------------------------------------------
# Satellite dynamics and simulation functions (unchanged)
# ------------------------------------------------------------------------------
def satellite_dynamics(t, state, params):
    """
    Compute satellite dynamics including orbital propagation, attitude dynamics,
    magnetic field, and solar vector computation.

    State vector structure:
      state[0]   : True anomaly (rad)
      state[1:5] : Quaternion (body with respect to orbit)
      state[5:8] : Angular velocity in body frame (rad/s)
    """
    # Unpack state
    true_anomaly = state[0]
    quat_body_orbit = state[1:5]
    omega_body = state[5:8]

    # Unpack simulation parameters
    init_datetime = params.get("init_datetime", None)  # initial datetime
    t0 = params["t_0"]  # initial time (seconds)
    ra = params["ra"]  # apogee radius
    rp = params["rp"]  # perigee radius
    arg_perigee = params["omega"]  # argument of perigee
    raan = params["OMEGA"]  # right ascension of the ascending node
    incl = params["i"]  # inclination (rad)
    mu = params["mu"]  # gravitational parameter
    J = params["J"]  # inertia matrix
    tau_p = params["tau_p_b"]  # disturbance torque
    kp = params.get("kp", 0.01)  # proportional gain
    kd = params.get("kd", 0.01)  # derivative gain
    omega_desired = params.get("w_od_d", np.zeros(3))
    quat_desired = params.get("q_od", np.array([1.0, 0.0, 0.0, 0.0]))

    # ----------------------------
    # Orbital dynamics computations
    # ----------------------------
    a = calculate_semimajor_axis(ra, rp)
    e = calculate_eccentricity(ra, rp)
    n = calculate_mean_motion(a, mu)
    true_anomaly_dot = calculate_true_anomaly_derivative(e, true_anomaly, n)

    M = n * (t - t0)
    E = calculate_eccentric_anomaly(M, e)
    R_pqw = calculate_rotation_matrix_from_inertial_to_pqw(arg_perigee, raan, incl)
    r_pqw = calculate_radius_vector_in_pqw(a, e, E)
    r_inertial = calculate_radius_vector_in_inertial(r_pqw, R_pqw)
    r_norm = np.linalg.norm(r_inertial)
    v_pqw = calculate_velocity_vector_in_pqw(a, e, n, r_norm, E)
    a_pqw = calculate_acceleration_vector_in_pqw(a, e, n, r_norm, E)
    v_inertial = calculate_velocity_vector_in_inertial(v_pqw, R_pqw)
    a_inertial = calculate_acceleration_vector_in_inertial(a_pqw, R_pqw)

    # ----------------------------
    # Orbit angular velocity calculations
    # ----------------------------
    omega_orbit = (
        calculate_angular_velocity_of_orbit_relative_to_inertial_referenced_in_inertial(
            r_inertial, v_inertial
        )
    )
    omega_orbit_dot = calculate_angular_acceleration_of_orbit_relative_to_inertial_referenced_in_inertial(
        r_inertial, v_inertial, a_inertial
    )

    # ----------------------------
    # Attitude dynamics
    # ----------------------------
    q_io = calculate_quaternion_from_orbital_parameters(
        arg_perigee, raan, incl, true_anomaly
    )
    R_i_o = calculate_rotation_matrix_from_quaternion(q_io)
    tau_control = pd_attitude_controller(
        quat_body_orbit, omega_body, quat_desired, omega_desired, kp, kd
    )
    quat_dot = quaternion_kinematics(quat_body_orbit, omega_body)
    omega_dot = attitude_dynamics(
        J,
        quat_body_orbit,
        omega_body,
        omega_orbit,
        omega_orbit_dot,
        R_i_o,
        tau_control,
        tau_p,
    )

    # ----------------------------
    # Magnetic field and solar vector computations
    # ----------------------------

    # Compute current date for IGRF (magnetic field) model
    current_date = init_datetime + timedelta(seconds=t)
    # Calculate magnetic field vectors: B_ENU (in ENU frame) and B_o (transformed to orbit frame)
    B_ENU, B_o = calculate_magnetic_field_in_orbit_frame(
        r_inertial, current_date, arg_perigee, true_anomaly, raan, incl, t
    )

    # Calculate the sun vector (advanced model) in the inertial frame and then transform to orbit frame.
    s_i = calculate_advanced_sun_vector_model(t, init_datetime)
    s_o = R_i_o.T @ np.array(s_i)

    # ----------------------------
    # Additional attitude representations for logging
    # ----------------------------
    quat_inertial_body = calculate_quaternion_product(q_io, quat_body_orbit)
    euler_ib = calculate_euler_angles_from_quaternion(quat_inertial_body)
    euler_io = calculate_euler_angles_from_quaternion(q_io)

    # ----------------------------
    # Log dictionary: record computed quantities.
    # ----------------------------
    log_entry = {
        "time": t,
        "theta": true_anomaly,
        "q_ob": quat_body_orbit,
        "w_ob_b": omega_body,
        "R_pqw_i": R_pqw,
        "q_ib": quat_inertial_body,
        "q_io": q_io,
        "R_i_o": R_i_o,
        "r_i": r_inertial,
        "v_i": v_inertial,
        "a_i": a_inertial,
        "w_io_i": omega_orbit,
        "euler_ib": euler_ib,
        "euler_io": euler_io,
        "B_ENU": B_ENU,
        "B_o": B_o,
        "s_o": s_o,
    }

    # Construct derivative of state vector
    state_dot = np.hstack((true_anomaly_dot, quat_dot, omega_dot))

    return state_dot, log_entry


def satellite_dynamics_wrapper(t, state, params):
    state_dot, _ = satellite_dynamics(t, state, params)
    return state_dot


def run_simulation(params, initial_state, t_span, n_steps):
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    result = solve_ivp(
        lambda t, state: satellite_dynamics_wrapper(t, state, params),
        t_span,
        initial_state,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        args=(params,),
    )

    keys = [
        "time",
        "theta",
        "q_ob",
        "w_ob_b",
        "R_pqw_i",
        "q_ib",
        "q_io",
        "R_i_o",
        "r_i",
        "v_i",
        "a_i",
        "w_io_i",
        "euler_ib",
        "euler_io",
    ]
    data_log = {key: [] for key in keys}

    for t_val, state in zip(result.t, result.y.T):
        _, log_entry = satellite_dynamics(t_val, state, params)
        for key in keys:
            data_log[key].append(log_entry[key])
    return result, data_log


def get_simulation_params(mu_value, earth_radius):
    return {
        "init_datetime": datetime(2022, 1, 1, 0, 0, 0),
        "t_0": 0,
        "ra": earth_radius + 10000,
        "rp": earth_radius + 400,
        "omega": np.radians(0),
        "OMEGA": np.radians(0),
        "i": np.radians(0),
        "mu": mu_value,
        "J": np.eye(3),
        "tau_p_b": np.zeros(3),
        "kp": 0.1,
        "kd": 0.1,
        "q_od": np.array([0, 0, 1, 0]),
        "w_od_d": np.radians([0.0, 0.0, 0.0]),
    }


# ------------------------------------------------------------------------------
# New functions for plotting sun vectors using the advanced model
# ------------------------------------------------------------------------------


def plot_sun_vectors_over_orbit(result, data_log, init_datetime):
    """
    For each time step of the satellite simulation (covering ~one orbit),
    compute the sun vector in the inertial frame using the advanced model
    and its representation in the orbit frame.

    Then plot the three components versus time.
    """
    t_array = result.t  # time in seconds
    inertial_sun = []
    orbit_sun = []

    # Loop over simulation time steps
    for i, t in enumerate(t_array):
        # Use the advanced sun vector model.
        # t is in seconds; init_datetime is passed from params.
        s_i = calculate_advanced_sun_vector_model(t, init_datetime)
        inertial_sun.append(s_i)
        # data_log["R_i_o"] is the rotation matrix from the orbit frame to the inertial frame.
        # To express the inertial sun vector in the orbit frame, apply the inverse rotation (i.e., transpose).
        R_i_o = data_log["R_i_o"][i]
        s_o = R_i_o.T @ np.array(s_i)
        orbit_sun.append(s_o)

    inertial_sun = np.array(inertial_sun)
    orbit_sun = np.array(orbit_sun)

    # Plot the vector components versus time (time converted to hours for clarity)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t_array / 3600, inertial_sun[:, 0], label="Sx")
    axs[0].plot(t_array / 3600, inertial_sun[:, 1], label="Sy")
    axs[0].plot(t_array / 3600, inertial_sun[:, 2], label="Sz")
    axs[0].set_ylabel("Inertial Sun Vector")
    axs[0].set_title("Advanced Sun Vector in Inertial Frame (over one orbit)")
    axs[0].legend()

    axs[1].plot(t_array / 3600, orbit_sun[:, 0], label="Sx")
    axs[1].plot(t_array / 3600, orbit_sun[:, 1], label="Sy")
    axs[1].plot(t_array / 3600, orbit_sun[:, 2], label="Sz")
    axs[1].set_xlabel("Time (hours)")
    axs[1].set_ylabel("Orbit Frame Sun Vector")
    axs[1].set_title("Advanced Sun Vector in Orbit Frame (over one orbit)")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"{directory_path}/sun_vector_inertial_vs_orbit.png")
    plt.show()

    # Explanation printed to the console:
    print("\nExplanation:")
    print(
        "In the inertial frame, the advanced sun vector (expressed in AU) changes only slightly over one orbit,"
    )
    print(
        "because the relative position of the sun changes negligibly during a short orbital period."
    )
    print(
        "However, the orbit frame rotates with the satellite, so the sun vector expressed in that frame"
    )
    print("appears time-varying.")


def plot_yearly_sun_vector(init_datetime):
    """
    Compute and plot the sun vector in the inertial frame over one year using the advanced model.
    The simulation is run over 365.24 days (converted to seconds).
    """
    # One year in seconds
    t_seconds = np.linspace(0, 365.24 * 86400, 1000)
    sun_vectors = np.array(
        [calculate_advanced_sun_vector_model(t, init_datetime) for t in t_seconds]
    )

    # Convert time axis to days for plotting
    t_days = t_seconds / 86400.0

    plt.figure(figsize=(10, 4))
    plt.plot(t_days, sun_vectors[:, 0], label="Sx")
    plt.plot(t_days, sun_vectors[:, 1], label="Sy")
    plt.plot(t_days, sun_vectors[:, 2], label="Sz")
    plt.xlabel("Time (days)")
    plt.ylabel("Sun Vector Components (AU)")
    plt.title("Advanced Sun Vector in Inertial Frame Over One Year")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{directory_path}/sun_vector_inertial_one_year.png")
    plt.show()


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():
    """
    Main function to set up and execute the satellite simulation and to
    plot the sun vector both in the inertial and orbit frames using the advanced model.
    """
    mu_value = 398600.4418
    earth_radius = 6378
    params = get_simulation_params(mu_value, earth_radius)
    init_datetime = params["init_datetime"]

    # Define initial state: [true anomaly, quaternion, angular velocity]
    initial_theta = np.radians(0.0)
    initial_quat = np.array([0.0, 1.0, 0.0, 0.0])
    initial_omega = np.array([0.1, 0.2, -0.3])
    initial_state = np.concatenate(([initial_theta], initial_quat, initial_omega))

    # Run a simulation covering roughly one orbit (4 hours simulation)
    t_span = [0, 3600 * 4]  # seconds
    n_steps = 1000

    result, data_log = run_simulation(params, initial_state, t_span, n_steps)

    # Plot and compare the sun vector in the inertial and orbit frames over one orbit.
    plot_sun_vectors_over_orbit(result, data_log, init_datetime)

    # Additionally, compute and plot the sun vector in the inertial frame over one year.
    plot_yearly_sun_vector(init_datetime)

    # Existing visualization code for the satellite (choose static scene or animation)
    static_scene = False
    if static_scene:
        plotter = visualize_scene(data_log, time_index=100)
        plotter.show()
    else:
        filename = (
            f"{directory_path}"
            f"SatelliteAnimation_i{round(params['i'],2)}_omega{round(params['omega'],2)}"
            f"_OMEGA{round(params['OMEGA'],2)}_q{np.round(params['q_od'],2)}_"
            f"{round(t_span[1] / 3600,2)}h_{n_steps}steps.gif"
        )
        animate_satellite(result.t, data_log, filename)


if __name__ == "__main__":
    main()
