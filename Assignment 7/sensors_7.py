import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_line_sphere_intersection_point(p1, p2, p3, r):
    """
    Calculate the intersection point of the line defined by p1 and p2 with a sphere
    centered at p3 with radius r.

    Parameters:
        p1 (array-like): First point on the line.
        p2 (array-like): Second point on the line.
        p3 (array-like): Center of the sphere.
        r (float): Radius of the sphere.

    Returns:
        tuple:
            - Intersection point (ndarray) if an intersection exists (or the tangent point).
            - Boolean flag indicating whether an intersection was found.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)

    d = p2 - p1  # Direction vector of the line
    f = p1 - p3  # Vector from sphere center to p1
    m = (p1 + p2) / 2  # Midpoint of the line segment

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None, False
    elif np.isclose(discriminant, 0.0):
        i = p1 + (-b / (2 * a)) * d
        return i, True
    else:
        sqrt_disc = np.sqrt(discriminant)
        i1 = p1 + ((-b + sqrt_disc) / (2 * a)) * d
        i2 = p1 + ((-b - sqrt_disc) / (2 * a)) * d
        # Return the intersection point closer to the midpoint of the line
        if np.linalg.norm(i1 - m) < np.linalg.norm(i2 - m):
            return i1, True
        else:
            return i2, True


def calculate_raycasting_points(
    R_i_b,
    r_i,
    raycasting_length=10000,
    field_of_view_half=np.radians(15),
    number_of_raycasting_points=10,
):
    """
    Generate raycasting points in the inertial frame based on the field of view.

    Parameters:
        R_i_b (ndarray): Rotation matrix from body to inertial frame.
        r_i (ndarray): Satellite position in inertial frame.
        raycasting_length (float): Distance along the body x-axis for raycasting.
        field_of_view_half (float): Half the field of view (in radians).
        number_of_raycasting_points (int): Number of points to generate along the circle.

    Returns:
        ndarray: Array of raycasting points in the inertial frame.
    """
    raycasting_points = []
    radius = raycasting_length * np.tan(field_of_view_half)

    for theta in np.linspace(0, 2 * np.pi, number_of_raycasting_points, endpoint=False):
        point_body = np.array(
            [raycasting_length, radius * np.cos(theta), radius * np.sin(theta)]
        )
        point_inertial = R_i_b @ point_body + r_i
        raycasting_points.append(point_inertial)

    return np.array(raycasting_points)


def calculate_intersection_points_in_inertial_frame(
    r_i,
    R_i_b,
    raycasting_length,
    field_of_view_half_deg,
    number_of_raycasting_points,
    radius,
):
    """
    Calculate the intersection points between raycasting lines from the satellite and a sphere.

    Parameters:
        r_i (ndarray): Satellite position in inertial frame.
        R_i_b (ndarray): Rotation matrix from body to inertial frame.
        raycasting_length (float): Length of the raycasting lines.
        field_of_view_half_deg (float): Half field of view in degrees.
        number_of_raycasting_points (int): Number of raycasting points.
        radius (float): Radius of the sphere.

    Returns:
        ndarray: Intersection points in the inertial frame.
    """
    fov_half_rad = np.radians(field_of_view_half_deg)
    raycasting_points = calculate_raycasting_points(
        R_i_b,
        r_i,
        raycasting_length=raycasting_length,
        field_of_view_half=fov_half_rad,
        number_of_raycasting_points=number_of_raycasting_points,
    )

    p1 = r_i
    p3 = np.array([0, 0, 0])  # Sphere center (e.g., Earth at the origin)
    intersection_points = np.zeros_like(raycasting_points)

    for idx, p2 in enumerate(raycasting_points):
        intersection, intersects = calculate_line_sphere_intersection_point(
            p1, p2, p3, radius
        )
        if intersects:
            intersection_points[idx] = intersection
        else:
            intersection_points[idx] = p2

    return intersection_points


# Example usage
if __name__ == "__main__":
    # Test the line-sphere intersection function
    p1 = [1, 2, 3]
    p2 = [4, 5, 6]
    p3 = [0, 0, 0]
    r = 5

    intersection_point, exists_intersection = calculate_line_sphere_intersection_point(
        p1, p2, p3, r
    )
    print("Intersection point:", intersection_point)
