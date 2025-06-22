import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Custom cumulative trapezoidal integrator to replace scipy.integrate.cumtrapz
def cumtrapz_custom(y, x, initial=0):
    """
    Compute the cumulative integral of y with respect to x using the trapezoidal rule.
    Parameters:
      y: array-like, the function values.
      x: array-like, the sample points corresponding to y.
      initial: float, initial value of the integration.
    Returns:
      cumulative integral array of the same length as y.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    n = len(y)
    result = np.zeros(n)
    result[0] = initial
    for i in range(1, n):
        result[i] = result[i - 1] + (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return result


# Import your simulation and visualization functions
from visualization_8 import visualize_scene, animate_satellite
from orbital_mechanics_8 import (
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
from attitude_dynamics_8 import (
    quaternion_kinematics,
    attitude_dynamics,
    calculate_euler_angles_from_quaternion,
    calculate_rotation_matrix_from_quaternion,
    calculate_quaternion_product,
    calculate_inverse_quaternion,
)
from controller_8 import pd_attitude_controller
from magnetic_field_8 import calculate_magnetic_field_in_orbit_frame
from sun_vector_models_8 import (
    calculate_advanced_sun_vector_model,
    rotation_matrix_and_euler,
)
from quest_algorithm_8 import (
    quest_algorithm,
    quaternion_sign_correction,
    match_quaternion_sign_to_reference,
    measure_sensor,
)
from actuators import attitude_control_using_thrusters

np.set_printoptions(precision=3, suppress=True)

directory_path = "Assignment 8/"


# ------------------------------------------------------------------------------
# Satellite dynamics and simulation functions (unchanged structure but with fixes)
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
    q_ob = state[1:5]
    omega_b = state[5:8]

    # Unpack simulation parameters
    init_datetime = params["init_datetime"]  # initial datetime
    t0 = params["t_0"]  # initial time (seconds)
    ra = params["ra"]  # apogee radius
    rp = params["rp"]  # perigee radius
    arg_perigee = params["omega"]  # argument of perigee
    raan = params["OMEGA"]  # right ascension of the ascending node
    incl = params["i"]  # inclination (rad)
    mu = params["mu"]  # gravitational parameter
    J = params["J"]  # inertia matrix
    tau_p = params["tau_p_b"]  # disturbance torque
    kp = params["kp"]  # proportional gain
    kd = params["kd"]  # derivative gain
    omega_desired = params["w_od_d"]
    quat_desired = params["q_od"]
    iteration = params["iteration"]

    # DEBUG: Only print every 1000 iterations:
    if iteration % 1000 == 0:
        print(f"Iteration: {iteration}, t = {t:.2f}")

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
    # Attitude representation conversions
    # ----------------------------
    q_io = calculate_quaternion_from_orbital_parameters(
        arg_perigee, raan, incl, true_anomaly
    )
    R_i_o = calculate_rotation_matrix_from_quaternion(q_io)

    # Compute body-to-orbit rotation matrix from the current attitude (q_ob)
    R_b_o = calculate_rotation_matrix_from_quaternion(q_ob).T

    # ----------------------------
    # Orbit angular velocity calculations
    # ----------------------------
    omega_o = (
        calculate_angular_velocity_of_orbit_relative_to_inertial_referenced_in_inertial(
            r_inertial, v_inertial
        )
    )
    omega_o_dot = calculate_angular_acceleration_of_orbit_relative_to_inertial_referenced_in_inertial(
        r_inertial, v_inertial, a_inertial
    )

    # ----------------------------
    # Solar vector and Magnetic Field vector
    # ----------------------------
    current_date = init_datetime + timedelta(seconds=t)
    # Magnetic field measurement
    B_ENU, B_o = calculate_magnetic_field_in_orbit_frame(
        r_inertial, current_date, arg_perigee, true_anomaly, raan, incl, t
    )
    B_b = R_b_o @ B_o
    B_b_mesured = measure_sensor(B_b, np.radians(5))
    # Solar vector measurement
    s_i = calculate_advanced_sun_vector_model(t, init_datetime)
    s_o = R_i_o.T @ np.array(s_i)
    s_b = R_b_o @ s_o
    s_b_mesured = measure_sensor(s_b, np.radians(0.01))

    # ----------------------------
    # Attitude Dynamics
    # ----------------------------
    # Approximation with QUEST algorithm
    q_ob_hat = quest_algorithm(B_b_mesured, B_o, s_b_mesured, s_o)
    q_ob_hat = quaternion_sign_correction(q_ob_hat, params["q_ob_hat_prev"])
    q_ob_hat = match_quaternion_sign_to_reference(q_ob_hat, params["q_ob_hat_prev"])

    # Controller to compute control torque
    tau_d_b = pd_attitude_controller(q_ob, omega_b, quat_desired, omega_desired, kp, kd)
    tau_a_b, thruster_firings = attitude_control_using_thrusters(
        tau_d_b, max_thrust=0.5
    )
    q_ob_dot = quaternion_kinematics(q_ob, omega_b)
    omega_dot = attitude_dynamics(
        J,
        q_ob,
        omega_b,
        omega_o,
        omega_o_dot,
        R_i_o,
        tau_a_b,
        tau_p,
    )

    # ----------------------------
    # Additional attitude representations for logging
    # ----------------------------
    q_ib = calculate_quaternion_product(q_io, q_ob)
    euler_ib = calculate_euler_angles_from_quaternion(q_ob)
    euler_io = calculate_euler_angles_from_quaternion(q_io)

    # ----------------------------
    # Update params dictionary for the next iteration
    # ----------------------------
    params["q_ob_hat_prev"] = q_ob_hat
    params["iteration"] = iteration + 1

    # ----------------------------
    # Log dictionary: record computed quantities.
    # ----------------------------
    log_entry = {
        "time": t,
        "theta": true_anomaly,
        "q_ob": q_ob,
        "q_ob_hat": q_ob_hat,
        "R_b_o": R_b_o,
        "w_ob_b": omega_b,
        "R_pqw_i": R_pqw,
        "q_ib": q_ib,
        "q_io": q_io,
        "R_i_o": R_i_o,
        "r_i": r_inertial,
        "v_i": v_inertial,
        "a_i": a_inertial,
        "w_io_i": omega_o,
        "euler_ib": euler_ib,
        "euler_io": euler_io,
        "B_ENU": B_ENU,
        "B_o": B_o,
        "s_o": s_o,
        "tau_d_b": tau_d_b,
        "tau_a_b": tau_a_b,
        "truster_firings": thruster_firings,
    }

    # Construct derivative of state vector
    state_dot = np.hstack((true_anomaly_dot, q_ob_dot, omega_dot))

    return state_dot, log_entry


def satellite_dynamics_wrapper(t, state, params):
    state_dot, _ = satellite_dynamics(t, state, params)
    return state_dot


def run_simulation(params, initial_state, t_span, n_steps):
    result = solve_ivp(
        satellite_dynamics_wrapper,
        t_span,
        initial_state,
        method="RK45",
        t_eval=np.linspace(t_span[0], t_span[1], n_steps),
        args=(params,),
    )
    t = result.t
    state_vector = result.y

    # Build the data log dictionary from the logged entries.
    keys = [
        "time",
        "theta",
        "q_ob",
        "q_ob_hat",
        "R_b_o",
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
        "B_ENU",
        "B_o",
        "s_o",
        "tau_d_b",
        "tau_a_b",
        "truster_firings",
    ]
    data_log = {key: [] for key in keys}

    # Populate the data log from each logged entry.
    for i in range(len(t)):
        time_val = t[i]
        state = state_vector[:, i]
        state_dot, log_entry = satellite_dynamics(time_val, state, params)

        for key in keys:
            data_log[key].append(log_entry[key])

    return result, data_log


def get_simulation_params(mu_value, earth_radius, q_ob_init):
    return {
        "iteration": 0,
        "init_datetime": datetime(2025, 2, 3, 9, 30, 5),
        "t_0": 0,
        "ra": earth_radius + 10000,
        "rp": earth_radius + 400,
        "omega": np.radians(0),
        "OMEGA": np.radians(0),
        "i": np.radians(0),
        "mu": mu_value,
        "J": 3 * np.eye(3),
        "tau_p_b": np.zeros(3),
        "kp": 10,
        "kd": 20,
        "q_od": np.array([1, 0, 0, 0]),
        "w_od_d": np.radians([0.0, 0.0, 0.0]),
        "q_ob_hat_prev": np.array(q_ob_init),
    }


# ------------------------------------------------------------------------------
# Plotting Functions:
# ------------------------------------------------------------------------------


def plot_quaternion_evolution(filename, quaternion_log, time_vector=None):
    """
    Plots the evolution of the quaternion components over time.
    """
    quaternion_log = np.array(quaternion_log)
    n_steps = quaternion_log.shape[0]
    if time_vector is None:
        time_vector = np.arange(n_steps)

    plt.figure(figsize=(10, 6))
    plt.plot(time_vector, quaternion_log[:, 0], label="$q_0$")
    plt.plot(time_vector, quaternion_log[:, 1], label="$q_1$")
    plt.plot(time_vector, quaternion_log[:, 2], label="$q_2$")
    plt.plot(time_vector, quaternion_log[:, 3], label="$q_3$")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Quaternion Component Value")
    plt.title("Quaternion Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_thruster_firings(filename, thruster_firings_log, time_vector=None):
    """
    Plots the thruster firing commands over time.
    Each thruster's bang-bang (with deadzone) command is plotted.
    """
    thruster_firings_log = np.array(thruster_firings_log)
    n_steps = thruster_firings_log.shape[0]
    if time_vector is None:
        time_vector = np.arange(n_steps)

    plt.figure(figsize=(10, 6))
    n_thrusters = thruster_firings_log.shape[1]
    for i in range(n_thrusters):
        plt.plot(time_vector, thruster_firings_log[:, i], label=f"Thruster {i+1}")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Thruster Firing [N]")
    plt.title("Thruster Firings Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_propellant_consumption(
    filename, thruster_firings_log, time_vector, Isp=250, g=9.81
):
    """
    Computes and plots the cumulative propellant consumption over time.
    The instantaneous propellant flow rate is calculated as:
        mdot = (sum of thruster forces) / (Isp * g)
    and then integrated over time.
    """
    thruster_firings_log = np.array(thruster_firings_log)
    total_thrust = np.sum(thruster_firings_log, axis=1)
    mdot = total_thrust / (Isp * g)  # kg/s

    # Compute cumulative propellant consumption using trapezoidal integration.
    m_prop = cumtrapz_custom(mdot, time_vector, initial=0)

    plt.figure(figsize=(10, 6))
    plt.plot(time_vector, m_prop, "b-", label="Propellant Consumed")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Propellant Consumption [$kg$]")
    plt.title("Cumulative Propellant Consumption")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():

    # Constants
    mu_value = 398600.4418
    earth_radius = 6378

    # Initial State Parameters
    q_ob_init = np.array([0.0, 0.0, 0.0, 1])
    initial_theta = np.radians(0.0)
    w_ob_b_init = np.array([0.0, 0.0, 0.0])

    # Parametres and initial state
    params = get_simulation_params(mu_value, earth_radius, q_ob_init)
    initial_state = np.hstack((initial_theta, q_ob_init, w_ob_b_init))

    # Run a simulation covering roughly one orbit (4 hours simulation)
    t_span = [0, 100]  # seconds
    n_steps = 500

    result, data_log = run_simulation(params, initial_state, t_span, n_steps)

    # Plot quaternion evolution using the logged data
    plot_quaternion_evolution(
        filename=f"{directory_path}quaternion_evolution.png",
        quaternion_log=data_log["q_ob"],
        time_vector=data_log["time"],
    )

    # Plot thruster firings evolution
    plot_thruster_firings(
        filename=f"{directory_path}thruster_firings.png",
        thruster_firings_log=data_log["truster_firings"],
        time_vector=data_log["time"],
    )

    # Plot cumulative propellant consumption using Isp=250 seconds
    plot_propellant_consumption(
        filename=f"{directory_path}propellant_consumption.png",
        thruster_firings_log=data_log["truster_firings"],
        time_vector=data_log["time"],
        Isp=250,
        g=9.81,
    )

    # Existing visualization code for the satellite (choose static scene or animation)
    static_scene = True
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
