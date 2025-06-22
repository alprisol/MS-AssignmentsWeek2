import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Custom cumulative trapezoidal integrator to replace scipy.integrate.cumtrapz
def cumtrapz_custom(y, x, initial=0):
    """
    Compute the cumulative integral of y with respect to x using the trapezoidal rule.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    n = len(y)
    result = np.zeros(n)
    result[0] = initial
    for i in range(1, n):
        result[i] = result[i - 1] + (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return result


# -------------------------------------------------------------------------------
# Import simulation and visualization functions
# -------------------------------------------------------------------------------
from visualization_9 import visualize_scene, animate_satellite
from orbital_mechanics_9 import (
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
from attitude_dynamics_9 import (
    quaternion_kinematics,
    attitude_dynamics,
    calculate_euler_angles_from_quaternion,
    calculate_rotation_matrix_from_quaternion,
    calculate_quaternion_product,
    calculate_inverse_quaternion,
)
from controller_9 import pd_attitude_controller
from magnetic_field_9 import calculate_magnetic_field_in_orbit_frame
from sun_vector_models_9 import (
    calculate_advanced_sun_vector_model,
    rotation_matrix_and_euler,
)
from quest_algorithm_9 import (
    quest_algorithm,
    quaternion_sign_correction,
    match_quaternion_sign_to_reference,
    measure_sensor,
)
from actuators_9 import (
    attitude_control_using_thrusters,
    attitude_control_using_magnetic_torquers,  # Modified function returns (tau_a_b, coil_currents)
    attitude_control_using_reaction_wheels_in_tethrahedron,
)

np.set_printoptions(precision=3, suppress=True)
directory_path = "Assignment 9/"


# -------------------------------------------------------------------------------
# Satellite dynamics and simulation functions
# -------------------------------------------------------------------------------
def satellite_dynamics(t, state, params):
    """
    Compute satellite dynamics including orbital propagation, attitude dynamics,
    and actuation.
    """
    # Unpack state vector and parameters
    actuator = params.get("actuator", "thrusters")
    true_anomaly = state[0]
    q_ob = state[1:5]
    omega_b = state[5:8]
    if actuator == "reaction_wheel":
        w_bw = state[8:12]  # Reaction wheel speeds

    # Unpack simulation parameters
    init_datetime = params["init_datetime"]
    t0 = params["t_0"]
    ra = params["ra"]
    rp = params["rp"]
    arg_perigee = params["omega"]
    raan = params["OMEGA"]
    incl = params["i"]
    mu = params["mu"]
    J = params["J"]
    tau_p = params["tau_p_b"]
    kp = params["kp"]
    kd = params["kd"]
    omega_desired = params["w_od_d"]
    quat_desired = params["q_od"]
    iteration = params["iteration"]
    if actuator == "magnetic_torquer":
        N = params["N"]
        A = params["A"]
        i_max = params["i_max"]

    # Debug print (only every 1000 iterations)
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
    R_b_o = calculate_rotation_matrix_from_quaternion(q_ob).T

    # ----------------------------
    # Orbit angular velocity computations
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
    # Sensor measurements (Magnetic field and Sun vector)
    # ----------------------------
    current_date = init_datetime + timedelta(seconds=t)
    B_ENU, B_o = calculate_magnetic_field_in_orbit_frame(
        r_inertial, current_date, arg_perigee, true_anomaly, raan, incl, t
    )
    B_ENU, B_o = B_ENU * 1e9, B_o * 1e9  # Convert to nT
    B_b = R_b_o @ B_o
    B_b_mesured = measure_sensor(B_b, np.radians(5))
    s_i = calculate_advanced_sun_vector_model(t, init_datetime)
    s_i = np.array(s_i) * 149597870.7  # Convert from AU to km
    s_o = R_i_o.T @ np.array(s_i)
    s_b = R_b_o @ s_o
    s_b_mesured = measure_sensor(s_b, np.radians(0.01))

    # ----------------------------
    # Attitude estimation using QUEST
    # ----------------------------
    q_ob_hat = quest_algorithm(B_b_mesured, B_o, s_b_mesured, s_o)
    q_ob_hat = quaternion_sign_correction(q_ob_hat, params["q_ob_hat_prev"])
    q_ob_hat = match_quaternion_sign_to_reference(q_ob_hat, params["q_ob_hat_prev"])

    # ----------------------------
    # Attitude control computation (PD controller)
    # ----------------------------
    tau_d_b = pd_attitude_controller(q_ob, omega_b, quat_desired, omega_desired, kp, kd)

    # ----------------------------
    # Actuator selection and computation
    # ----------------------------
    if actuator == "thrusters":
        tau_a_b, thruster_firings = attitude_control_using_thrusters(
            tau_d_b, max_thrust=0.5
        )
    elif actuator == "magnetic_torquer":
        # Now capture both the actuation torque and the coil currents.
        tau_a_b, coil_currents = attitude_control_using_magnetic_torquers(
            tau_d_b, B_b, A, N, i_max
        )
        thruster_firings = None
    elif actuator == "reaction_wheel":
        w_bw = state[8:12]
        w_bw_dot, tau_a_b = attitude_control_using_reaction_wheels_in_tethrahedron(
            tau_d_b, omega_b, w_bw, params["J_w"], params["max_RPM"]
        )
    else:
        raise ValueError(
            "Unknown actuator type. Choose thrusters, magnetic_torquer, or reaction_wheel."
        )

    # ----------------------------
    # Attitude kinematics and dynamics
    # ----------------------------
    q_ob_dot = quaternion_kinematics(q_ob, omega_b)
    omega_dot = attitude_dynamics(
        J, q_ob, omega_b, omega_o, omega_o_dot, R_i_o, tau_a_b, tau_p
    )

    # ----------------------------
    # Additional attitude representations for logging
    # ----------------------------
    q_ib = calculate_quaternion_product(q_io, q_ob_hat)
    euler_ib = calculate_euler_angles_from_quaternion(q_ob_hat)
    euler_io = calculate_euler_angles_from_quaternion(q_io)

    # Update parameters for next iteration
    params["q_ob_hat_prev"] = q_ob_hat
    params["iteration"] = iteration + 1

    # ----------------------------
    # Build log dictionary
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
    }
    # Log actuator-specific data
    if actuator == "thrusters":
        log_entry["thruster_firings"] = thruster_firings
    elif actuator == "reaction_wheel":
        log_entry["w_bw"] = w_bw
    elif actuator == "magnetic_torquer":
        log_entry["coil_currents"] = coil_currents

    # ----------------------------
    # Construct state derivative vector based on actuator type.
    # ----------------------------
    if actuator == "reaction_wheel":
        state_dot = np.hstack((true_anomaly_dot, q_ob_dot, omega_dot, w_bw_dot))
    else:
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

    # Choose keys based on actuator type.
    actuator = params["actuator"]
    if actuator == "reaction_wheel":
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
            "w_bw",
        ]
    elif actuator == "thrusters":
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
            "thruster_firings",
        ]
    else:  # magnetic torquer
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
            "coil_currents",
        ]

    data_log = {key: [] for key in keys}
    for i in range(len(t)):
        time_val = t[i]
        state_i = state_vector[:, i]
        _, log_entry = satellite_dynamics(time_val, state_i, params)
        for key in keys:
            data_log[key].append(log_entry[key])
    return result, data_log


def get_simulation_params(mu_value, earth_radius, q_ob_init, actuator):
    """
    Set simulation parameters.
    """
    params = {
        "iteration": 0,
        "init_datetime": datetime(2025, 2, 3, 9, 30, 5),
        "t_0": 0,
        "ra": earth_radius + 400,
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
        "actuator": actuator,
        "N": 500,
        "A": 1.0,
        "i_max": 0.1,
    }
    if params["actuator"] == "reaction_wheel":
        params["J_w"] = 0.1
        params["max_RPM"] = 20000
    return params


# -------------------------------------------------------------------------------
# Plotting Functions
# -------------------------------------------------------------------------------
def plot_quaternion_evolution(
    filename, quaternion_log, time_vector=None, title="Quaternion Evolution"
):
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
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_attitude_evolution(
    filename, attitude_log, time_vector=None, title="Attitude Evolution (Euler Angles)"
):
    attitude_log = np.array(attitude_log)
    n_steps = attitude_log.shape[0]
    if time_vector is None:
        time_vector = np.arange(n_steps)
    plt.figure(figsize=(10, 6))
    plt.plot(time_vector, attitude_log[:, 0], label="Roll")
    plt.plot(time_vector, attitude_log[:, 1], label="Pitch")
    plt.plot(time_vector, attitude_log[:, 2], label="Yaw")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Angle ($rad$)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_reaction_wheel_speeds(
    filename, time_vector, reaction_wheel_log, title="Reaction Wheel Speeds Over Time"
):
    reaction_wheel_log = np.array(reaction_wheel_log)
    plt.figure(figsize=(8, 5))
    for i in range(reaction_wheel_log.shape[1]):
        plt.plot(time_vector, reaction_wheel_log[:, i], label=f"Wheel {i+1}")
    plt.xlabel("Time ($s$)")
    plt.ylabel("Reaction Wheel Speed ($rad/s$)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_thruster_firings(
    filename, thruster_firings_log, time_vector=None, title="Thruster Firings Over Time"
):
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
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_propellant_consumption(
    filename,
    thruster_firings_log,
    time_vector,
    Isp=250,
    g=9.81,
    title="Cumulative Propellant Consumption",
):
    thruster_firings_log = np.array(thruster_firings_log)
    total_thrust = np.sum(thruster_firings_log, axis=1)
    mdot = total_thrust / (Isp * g)
    m_prop = cumtrapz_custom(mdot, time_vector, initial=0)
    plt.figure(figsize=(10, 6))
    plt.plot(time_vector, m_prop, "b-", label="Propellant Consumed")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Propellant Consumption [$kg$]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# --- New Plot Function for Coil Currents ---
def plot_coil_currents(
    filename, time_vector, coil_currents_log, title="Coil Currents Over Time"
):
    """
    Plots the coil currents (one curve per coil) over time.
    """
    coil_currents_log = np.array(coil_currents_log)
    plt.figure(figsize=(8, 5))
    for i in range(coil_currents_log.shape[1]):
        plt.plot(time_vector, coil_currents_log[:, i], label=f"Coil {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# -------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------
def main():

    # Set actuator type here. Choose between "thrusters", "reaction_wheel", or "magnetic_torquer".
    actuator = "thrusters"

    # Constants
    mu_value = 398600.4418
    earth_radius = 6378

    # Initial State Parameters
    q_ob_init = np.array([0.0, 0.0, 0.0, 1.0])
    initial_theta = np.radians(0.0)
    w_ob_b_init = np.array([0.0, 0.0, 0.0])
    w_bw0 = (2 * np.pi / 60) * np.array([2000, -3000, 5000, 1000])

    # Get simulation parameters.
    params = get_simulation_params(mu_value, earth_radius, q_ob_init, actuator)

    # Initial state based on actuator type.
    if params.get("actuator") == "reaction_wheel":
        initial_state = np.hstack((initial_theta, q_ob_init, w_ob_b_init, w_bw0))
    else:
        initial_state = np.hstack((initial_theta, q_ob_init, w_ob_b_init))

    # Run simulation: adjust t_span and n_steps as needed.
    t_span = [0, 100]  # seconds
    n_steps = 500

    result, data_log = run_simulation(params, initial_state, t_span, n_steps)

    # Plot quaternion evolution with custom title.
    plot_quaternion_evolution(
        filename=f"{directory_path}quaternion_evolution_{actuator}.png",
        quaternion_log=data_log["q_ob"],
        time_vector=data_log["time"],
        title=f"{actuator.capitalize()} - Quaternion Evolution",
    )

    # Plot actuator-specific data.
    if params["actuator"] == "thrusters":
        plot_thruster_firings(
            filename=f"{directory_path}thruster_firings.png",
            thruster_firings_log=data_log["thruster_firings"],
            time_vector=data_log["time"],
            title=f"{actuator.capitalize()} - Thruster Firings",
        )
        plot_propellant_consumption(
            filename=f"{directory_path}propellant_consumption.png",
            thruster_firings_log=data_log["thruster_firings"],
            time_vector=data_log["time"],
            Isp=250,
            g=9.81,
            title=f"{actuator.capitalize()} - Propellant Consumption",
        )
    elif params["actuator"] == "reaction_wheel":
        plot_reaction_wheel_speeds(
            filename=f"{directory_path}reaction_wheel_speeds.png",
            time_vector=data_log["time"],
            reaction_wheel_log=data_log["w_bw"],
            title=f"{actuator.capitalize()} - Reaction Wheel Speeds",
        )
    elif params["actuator"] == "magnetic_torquer":
        # --- New plotting for coil currents ---
        plot_coil_currents(
            filename=f"{directory_path}coil_currents.png",
            time_vector=data_log["time"],
            coil_currents_log=data_log["coil_currents"],
            title=f"{actuator.capitalize()} - Coil Currents Over Time",
        )

    # Visualization (choose static scene or animation)
    static_scene = True
    if static_scene:
        plotter = visualize_scene(data_log, time_index=100)
        plotter.show()
    else:
        filename = (
            f"{directory_path}SatelliteAnimation_i{round(params['i'],2)}"
            f"_omega{round(params['omega'],2)}_OMEGA{round(params['OMEGA'],2)}"
            f"_q{np.round(params['q_od'],2)}_{round(t_span[1] / 3600,2)}h_{n_steps}steps.gif"
        )
        animate_satellite(result.t, data_log, filename)


if __name__ == "__main__":
    main()
