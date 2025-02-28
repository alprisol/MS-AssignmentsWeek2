import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import vtk

from sensors_7 import calculate_intersection_points_in_inertial_frame

# Global settings
np.set_printoptions(precision=3, suppress=True)
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xcolor}"

earth_rot_vel = 7.2722e-5  # Earth's rotation rate in rad/s


# ---------------------------------------------------------------------
# Rotation Helper Functions
# ---------------------------------------------------------------------
def Rz(angle):
    """
    Return the rotation matrix about the z-axis.
    """
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def Ry(angle):
    """
    Return the rotation matrix about the y-axis.
    """
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def Rx(angle):
    """
    Return the rotation matrix about the x-axis.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def pyvista_rotation_matrix_from_euler_angles(orientation_euler, degrees=False):
    """
    Compute the 3x3 rotation matrix from Euler angles.
    Uses the y-x-z rotation order.

    Parameters:
        orientation_euler (list): [phi, theta, psi] angles.
        degrees (bool): Whether the angles are in degrees.

    Returns:
        ndarray: 3x3 rotation matrix.
    """
    if degrees:
        phi, theta, psi = orientation_euler
    else:
        phi, theta, psi = np.array(orientation_euler) * np.pi / 180
    return Rz(psi).dot(Rx(phi)).dot(Ry(theta))


# ---------------------------------------------------------------------
# Object Creation Functions
# ---------------------------------------------------------------------
def create_reference_frame(plotter, labels, scale=1):
    """
    Create a 3D reference frame with labeled axes.

    Parameters:
        plotter (pv.Plotter): PyVista plotter.
        labels (list): Labels for x, y, z axes.
        scale (float): Scale factor for arrow and label size.

    Returns:
        dict: Contains mesh objects and label objects.
    """
    x_arrow = pv.Arrow(
        start=(0, 0, 0),
        direction=(1, 0, 0),
        tip_length=0.25,
        tip_radius=0.1,
        tip_resolution=20,
        shaft_radius=0.05,
        shaft_resolution=20,
        scale=scale,
    )
    y_arrow = pv.Arrow(
        start=(0, 0, 0),
        direction=(0, 1, 0),
        tip_length=0.25,
        tip_radius=0.1,
        tip_resolution=20,
        shaft_radius=0.05,
        shaft_resolution=20,
        scale=scale,
    )
    z_arrow = pv.Arrow(
        start=(0, 0, 0),
        direction=(0, 0, 1),
        tip_length=0.25,
        tip_radius=0.1,
        tip_resolution=20,
        shaft_radius=0.05,
        shaft_resolution=20,
        scale=scale,
    )
    ref_frame = {
        "scale": scale,
        "x": plotter.add_mesh(x_arrow, color="red", show_edges=False),
        "y": plotter.add_mesh(y_arrow, color="blue", show_edges=False),
        "z": plotter.add_mesh(z_arrow, color="green", show_edges=False),
        "x_label": pv.Label(
            text=labels[0], position=np.array([1, 0, 0]) * scale, size=20
        ),
        "y_label": pv.Label(
            text=labels[1], position=np.array([0, 1, 0]) * scale, size=20
        ),
        "z_label": pv.Label(
            text=labels[2], position=np.array([0, 0, 1]) * scale, size=20
        ),
    }
    return ref_frame


def create_satellite(plotter, size=0.5):
    """
    Create a textured 3D satellite model with a body, solar panels, and a scientific instrument.

    Parameters:
        plotter (pv.Plotter): PyVista plotter.
        size (float): Scaling factor for the satellite size.

    Returns:
        dict: Satellite mesh objects.
    """
    # Satellite body
    body_texture = pv.read_texture("textures/satellite_texture.png")
    body = pv.Box(bounds=(-size, size, -size, size, -size, size))
    body.texture_map_to_plane(inplace=True)

    # Solar panels
    solar_panel_texture = pv.read_texture(
        "textures/high_quality_solar_panel_texture.png"
    )
    panel_width = 1.5 * size
    panel_length = 5 * size
    panel_thickness = 0.1 * size
    center_offset = 0.5 * size
    panel1 = pv.Box(
        bounds=(
            -panel_thickness / 2,
            panel_thickness / 2,
            -panel_width / 2,
            panel_width / 2,
            center_offset,
            panel_length,
        )
    )
    panel1.texture_map_to_plane(
        origin=(-size - panel_thickness / 2, 0, 0),
        point_u=(-size - panel_thickness / 2, panel_width / 2, 0),
        point_v=(-size - panel_thickness / 2, 0, panel_length / 2),
        inplace=True,
    )
    panel2 = pv.Box(
        bounds=(
            -panel_thickness / 2,
            panel_thickness / 2,
            -panel_width / 2,
            panel_width / 2,
            -center_offset,
            -panel_length,
        )
    )
    panel2.texture_map_to_plane(
        origin=(-size - panel_thickness / 2, 0, 0),
        point_u=(-size - panel_thickness / 2, panel_width / 2, 0),
        point_v=(-size - panel_thickness / 2, 0, panel_length / 2),
        inplace=True,
    )

    # Scientific instrument
    sci_texture = pv.read_texture("textures/camera_texture.png")
    instrument = pv.Cone(
        center=(size - 0.01, 0, 0),
        direction=(-1, 0, 0),
        height=0.5 * size,
        radius=0.5 * size,
        resolution=50,
    )
    instrument.texture_map_to_sphere(inplace=True)

    satellite = {
        "Body": plotter.add_mesh(body, texture=body_texture, show_edges=True),
        "Solar Panels": [
            plotter.add_mesh(panel1, texture=solar_panel_texture, show_edges=True),
            plotter.add_mesh(panel2, texture=solar_panel_texture, show_edges=True),
        ],
        "Scientific Instrument": plotter.add_mesh(
            instrument, texture=sci_texture, show_edges=False
        ),
    }
    return satellite


def create_earth(plotter, radius):
    """
    Create a 3D textured model of Earth.

    Parameters:
        plotter (pv.Plotter): PyVista plotter.
        radius (float): Earth model radius.

    Returns:
        The Earth mesh object.
    """
    earth = pv.examples.planets.load_earth(radius=radius)
    earth_texture = pv.examples.load_globe_texture()
    return plotter.add_mesh(earth, texture=earth_texture, smooth_shading=True)


def create_sensor_cone(
    plotter,
    r_i,
    R_i_b,
    raycasting_length,
    field_of_view_half_deg,
    number_of_raycasting_points,
    earth_radius,
):
    """
    Create a sensor cone mesh by intersecting raycasting lines with a sphere.

    Parameters:
        r_i (ndarray): Satellite position in inertial frame.
        R_i_b (ndarray): Rotation matrix from body to inertial frame.
        raycasting_length (float): Length of the raycasting rays.
        field_of_view_half_deg (float): Half field-of-view in degrees.
        number_of_raycasting_points (int): Number of rays.
        earth_radius (float): Sphere radius.

    Returns:
        The sensor cone mesh.
    """
    intersection_points = calculate_intersection_points_in_inertial_frame(
        r_i,
        R_i_b,
        raycasting_length,
        field_of_view_half_deg,
        number_of_raycasting_points,
        earth_radius,
    )
    triangles = []
    for i in range(len(intersection_points[:, 0]) - 1):
        triangles.append([r_i, intersection_points[i + 1], intersection_points[i]])
    triangles.append([r_i, intersection_points[0], intersection_points[-1]])
    cone_mesh = pv.PolyData()
    for tri in triangles:
        cone_mesh += pv.Triangle(tri)
    plotter.add_mesh(cone_mesh, opacity=0.25, color="green")
    return cone_mesh


# ---------------------------------------------------------------------
# Update Functions
# ---------------------------------------------------------------------
def update_satellite_pose(satellite_mesh, r_i, euler_ib, degrees=True):
    """
    Update the satellite's position and orientation.

    Parameters:
        satellite_mesh (dict): Satellite mesh objects.
        r_i (list): Satellite position in the inertial frame.
        euler_ib (list): Euler angles [phi, theta, psi] for body orientation.
        degrees (bool): Whether angles are in degrees.
    """
    if not degrees:
        euler_ib = np.rad2deg(euler_ib)
    satellite_mesh["Body"].SetPosition(r_i)
    for panel in satellite_mesh["Solar Panels"]:
        panel.SetPosition(r_i)
    satellite_mesh["Scientific Instrument"].SetPosition(r_i)
    satellite_mesh["Body"].SetOrientation(euler_ib)
    for panel in satellite_mesh["Solar Panels"]:
        panel.SetOrientation(euler_ib)
    satellite_mesh["Scientific Instrument"].SetOrientation(euler_ib)


def update_earth_orientation(earth_mesh, t):
    """
    Update Earth's orientation based on elapsed time.

    Parameters:
        earth_mesh: Earth mesh object.
        t (float): Time in seconds.
    """
    w_ie_deg = np.rad2deg(earth_rot_vel)
    orientation_deg = w_ie_deg * t
    earth_mesh.SetOrientation([0.0, 0.0, orientation_deg])


def update_sensor_cone_points(
    plotter,
    line_actor,
    sensor_cone_mesh,
    r_i,
    R_i_b,
    raycasting_length,
    field_of_view_half_deg,
    number_of_raycasting_points,
    earth_radius,
):
    """
    Update sensor cone intersection points and refresh the line actor.

    Parameters:
        plotter (pv.Plotter): PyVista plotter.
        line_actor: Existing line actor (or None).
        sensor_cone_mesh: Sensor cone mesh.
        r_i (array-like): Satellite position.
        R_i_b (ndarray): Rotation matrix.
        raycasting_length (float): Raycasting length.
        field_of_view_half_deg (float): Half FOV in degrees.
        number_of_raycasting_points (int): Number of rays.
        earth_radius (float): Sphere radius.

    Returns:
        Updated line actor.
    """
    intersection_points = calculate_intersection_points_in_inertial_frame(
        r_i,
        R_i_b,
        raycasting_length,
        field_of_view_half_deg,
        number_of_raycasting_points,
        earth_radius,
    )
    new_points = np.vstack([r_i, intersection_points])
    sensor_cone_mesh.points = new_points
    circle_lines = [
        [intersection_points[i], intersection_points[i + 1]]
        for i in range(len(intersection_points) - 1)
    ]
    circle_lines.append([intersection_points[-1], intersection_points[0]])
    line_points = np.array(circle_lines).reshape(-1, 3)
    if line_actor is not None:
        plotter.remove_actor(line_actor)
    line_actor = plotter.add_lines(line_points, color="green", width=2)
    return line_actor


def update_body_frame_pose(body_frame, r_i, euler_ib, degrees=True):
    """
    Update the body frame axes and label positions.

    Parameters:
        body_frame (dict): Contains mesh objects and labels.
        r_i (array-like): Position of the body frame.
        euler_ib (list): Euler angles for body orientation.
        degrees (bool): If angles are in degrees.
    """
    if not degrees:
        euler_ib = np.rad2deg(euler_ib)
    for key in ["x", "y", "z"]:
        body_frame[key].SetPosition(r_i)
        body_frame[key].SetOrientation(euler_ib)
    R_i_b = pyvista_rotation_matrix_from_euler_angles(euler_ib)
    scale = body_frame["scale"]
    body_frame["x_label"].position = r_i + R_i_b.dot([1, 0, 0]) * scale
    body_frame["y_label"].position = r_i + R_i_b.dot([0, 1, 0]) * scale
    body_frame["z_label"].position = r_i + R_i_b.dot([0, 0, 1]) * scale


def update_orbit_frame_pose(orbit_frame, r_i, euler_io, degrees=True):
    """
    Update the orbit frame axes and label positions.

    Parameters:
        orbit_frame (dict): Orbit frame mesh and labels.
        r_i (array-like): Position.
        euler_io (list): Euler angles for orbit orientation.
        degrees (bool): If angles are in degrees.
    """
    if not degrees:
        euler_io = np.rad2deg(euler_io)
    for key in ["x", "y", "z"]:
        orbit_frame[key].SetPosition(r_i)
        orbit_frame[key].SetOrientation(euler_io)
    R_i_o = pyvista_rotation_matrix_from_euler_angles(euler_io)
    scale = orbit_frame["scale"]
    orbit_frame["x_label"].position = r_i + R_i_o[:, 0] * scale
    orbit_frame["y_label"].position = r_i + R_i_o[:, 1] * scale
    orbit_frame["z_label"].position = r_i + R_i_o[:, 2] * scale


def update_ecef_frame_orientation(ecef_frame, t):
    """
    Update the ECEF frame orientation about its z-axis and reposition its labels.

    Parameters:
        ecef_frame (dict): ECEF frame mesh and labels.
        t (float): Time in seconds.
    """
    w_ie_deg = ecef_frame.get("w_ie_deg", np.rad2deg(earth_rot_vel))
    angle_z_deg = w_ie_deg * t
    for key in ["x", "y", "z"]:
        ecef_frame[key].SetOrientation([0.0, 0.0, angle_z_deg])
    angle_z_rad = np.deg2rad(angle_z_deg)
    RotZ = Rz(angle_z_rad)
    scale = ecef_frame["scale"]
    ecef_frame["x_label"].position = RotZ.dot([1, 0, 0]) * scale
    ecef_frame["y_label"].position = RotZ.dot([0, 1, 0]) * scale
    ecef_frame["z_label"].position = RotZ.dot([0, 0, 1]) * scale


# ---------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------
def visualize_scene(data_log, time_index=-1, off_screen=False):
    """
    Create a static 3D scene of the satellite around Earth with reference frames.

    Parameters:
        data_log (dict): Must contain keys "time", "r_i", "euler_ib", and "euler_io".
        time_index (int): Index of the snapshot to visualize.
        off_screen (bool): Use off-screen rendering if True.

    Returns:
        pv.Plotter: Configured plotter.
    """
    earth_radius = 6378
    raycasting_length = 10000
    field_of_view_half_deg = 15
    number_of_raycasting_points = 100
    frames_scale = [2, 1.5, 0.5]
    satellite_size = 0.1

    t_array = np.array(data_log["time"])
    r_i_array = np.array(data_log["r_i"])
    THETA_ib_array = np.array(data_log["euler_ib"])
    THETA_io_array = np.array(data_log["euler_io"])
    if time_index >= len(t_array):
        time_index = len(t_array) - 1

    time_value = t_array[time_index]
    r_i = r_i_array[time_index]
    euler_ib = THETA_ib_array[time_index]
    euler_io = THETA_io_array[time_index]

    plotter = pv.Plotter(off_screen=off_screen)

    satellite_mesh = create_satellite(plotter, size=satellite_size * earth_radius)
    earth_mesh = create_earth(plotter, radius=earth_radius)

    R_i_b = pyvista_rotation_matrix_from_euler_angles(euler_ib)
    line_actor = plotter.add_lines(np.empty((0, 3)), color="green", width=2)
    sensor_cone_mesh = create_sensor_cone(
        plotter,
        r_i,
        R_i_b,
        raycasting_length,
        field_of_view_half_deg,
        number_of_raycasting_points,
        earth_radius,
    )

    orbit_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^o$", "$\\mathbf{y}^o$", "$\\mathbf{z}^o$"],
        scale=frames_scale[2] * earth_radius,
    )
    eci_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^i$", "$\\mathbf{y}^i$", "$\\mathbf{z}^i$"],
        scale=frames_scale[0] * earth_radius,
    )
    ecef_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^e$", "$\\mathbf{y}^e$", "$\\mathbf{z}^e$"],
        scale=frames_scale[1] * earth_radius,
    )
    body_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^b$", "$\\mathbf{y}^b$", "$\\mathbf{z}^b$"],
        scale=frames_scale[2] * earth_radius,
    )

    for frame in [orbit_frame, eci_frame, ecef_frame, body_frame]:
        for key in ["x_label", "y_label", "z_label"]:
            plotter.add_actor(frame[key])

    update_satellite_pose(satellite_mesh, r_i, euler_ib, degrees=True)
    update_earth_orientation(earth_mesh, time_value)
    update_ecef_frame_orientation(ecef_frame, time_value)
    update_body_frame_pose(body_frame, r_i, euler_ib, degrees=True)
    update_orbit_frame_pose(orbit_frame, r_i, euler_io, degrees=True)
    line_actor = update_sensor_cone_points(
        plotter,
        line_actor,
        sensor_cone_mesh,
        r_i,
        R_i_b,
        raycasting_length,
        field_of_view_half_deg,
        number_of_raycasting_points,
        earth_radius,
    )
    plotter.camera.focal_point = (0, 0, 0)
    return plotter


def animate_satellite(t, data_log, gif_name):
    """
    Animate the satellite's position and orientation over time.

    Parameters:
        t (ndarray): Array of time values.
        data_log (dict): Logged satellite data.
        gif_name (str): Output GIF file name.
    """
    plotter = pv.Plotter(off_screen=False)
    earth_radius = 6378
    raycasting_length = 10000
    field_of_view_half_deg = 15
    number_of_raycasting_points = 100
    frames_scale = [2, 1.5, 0.5]
    satellite_size = 0.1

    satellite_mesh = create_satellite(plotter, size=satellite_size * earth_radius)
    earth_mesh = create_earth(plotter, radius=earth_radius)

    R_i_b = pyvista_rotation_matrix_from_euler_angles(data_log["euler_ib"][0])
    r_i = data_log["r_i"][0]
    sensor_cone_mesh = create_sensor_cone(
        plotter,
        r_i,
        R_i_b,
        raycasting_length,
        field_of_view_half_deg,
        number_of_raycasting_points,
        earth_radius,
    )
    line_actor = plotter.add_lines(np.empty((0, 3)), color="green", width=2)

    orbit_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^o$", "$\\mathbf{y}^o$", "$\\mathbf{z}^o$"],
        scale=frames_scale[2] * earth_radius,
    )
    eci_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^i$", "$\\mathbf{y}^i$", "$\\mathbf{z}^i$"],
        scale=frames_scale[0] * earth_radius,
    )
    ecef_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^e$", "$\\mathbf{y}^e$", "$\\mathbf{z}^e$"],
        scale=frames_scale[1] * earth_radius,
    )
    body_frame = create_reference_frame(
        plotter,
        labels=["$\\mathbf{x}^b$", "$\\mathbf{y}^b$", "$\\mathbf{z}^b$"],
        scale=frames_scale[2] * earth_radius,
    )
    for frame in [orbit_frame, eci_frame, ecef_frame, body_frame]:
        for key in ["x_label", "y_label", "z_label"]:
            plotter.add_actor(frame[key])

    r_i_array = np.array(data_log["r_i"])

    plotter.open_gif(gif_name)

    for i in range(len(t)):
        time_val = t[i]
        r_i = r_i_array[i]
        euler_ib = np.array(data_log["euler_ib"][i])
        euler_io = np.array(data_log["euler_io"][i])
        R_i_b = pyvista_rotation_matrix_from_euler_angles(euler_ib)

        update_satellite_pose(satellite_mesh, r_i, euler_ib)
        update_earth_orientation(earth_mesh, time_val)
        update_ecef_frame_orientation(ecef_frame, time_val)
        update_body_frame_pose(body_frame, r_i, euler_ib)
        update_orbit_frame_pose(orbit_frame, r_i, euler_io)
        line_actor = update_sensor_cone_points(
            plotter,
            line_actor,
            sensor_cone_mesh,
            r_i,
            R_i_b,
            raycasting_length,
            field_of_view_half_deg,
            number_of_raycasting_points,
            earth_radius,
        )
        plotter.write_frame()

    plotter.close()


if __name__ == "__main__":
    pass
