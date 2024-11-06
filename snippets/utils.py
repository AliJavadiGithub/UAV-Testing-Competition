"""
    Copied From TUMB at the SBFT 2024 Tool Competition - CPS-UAV Test Case Generation Track
    https://github.com/MayDGT/UAV-Testing-Competition
"""
import math
import random
import sys
import matplotlib.pyplot as pl
import matplotlib.patches as patches
import matplotlib as mpl

def random_rectangle(center_x, center_y, radius, eps=0.1):
    """Create random rectangle (x, y, l, w, r) inside a given circle."""
    min_length = eps * radius
    max_length = (1 - eps) * radius
    length = random.uniform(min_length, max_length) # half length, since radius is half diameter
    width = math.sqrt(radius ** 2 - length ** 2) # half width, since radius is half diameter
    rotation = random.uniform(0, 90) # in degrees
    scalar = 0.999 # make it a bit smaller, as we discussed
    return (center_x, center_y, scalar * length * 2, scalar * width * 2, rotation)

def get_subrectangles(x, y, l, w, r, count=3):
    """Subdivide the rectangle in x and y direction count times
        Name: get_subrectangles
        Parameters:
        x, y: Coordinates of the center of the rectangle.
        l, w: Length and width of the rectangle.
        r: Rotation angle of the rectangle in degrees.
        count: Number of subdivisions in both x and y directions (default: 3).
        Purpose:

        The function is designed to subdivide a given rectangle into smaller rectangles based on the specified parameters.
        The subdivision is performed in both the x and y directions, creating a grid-like structure of smaller rectangles.
        The rotation angle r allows the rectangle to be rotated before subdivision, ensuring that the resulting subrectangles are also rotated accordingly.
    """

    # Angle conversion to Radians
    r_radian = math.pi * r / 180

    # Rectangle Dimensions
    x1 = x - l / 2
    x2 = x + l / 2
    xmin = min([x1, x2])
    xmax = max([x1, x2])
    xdiff = xmax - xmin

    y1 = y - w / 2
    y2 = y + w / 2
    ymin = min([y1, y2])
    ymax = max([y1, y2])
    ydiff = ymax - ymin

    rectangles = []

    # Subdivision Loop
    for i in range(count):
        for j in range(count):

            # Subrectangle Calculation

            # Calculate the minimum x coordinate of the current subrectangle
            rxmin = xmin + i * xdiff / count

            # Calculate the x center coordinate of the current subrectangle
            rx = rxmin + xdiff / (2 * count)

            # Calculate the minimum y coordinate of the current subrectangle
            rymin = ymin + j * ydiff / count

            # Calculate the y center coordinate of the current subrectangle
            ry = rymin + (ydiff / (2 * count))

            # Calculate the length and width of the current subrectangle
            rl = l / count
            rw = w / count

            # Calculate the relative position of the subrectangle's center to the original rectangle's center
            dx = rx - x
            dy = ry - y

            # ensure that the subrectangle is rotated along with the original rectangle
            # Rotation matrix c, -s; s, c
            newdx = math.cos(r_radian) * dx - math.sin(r_radian) * dy
            newdy = math.sin(r_radian) * dx + math.cos(r_radian) * dy

            rectangles.append((x + newdx, y + newdy, rl, rw, r))

    return rectangles


def single_circle_coverage(x, y, l, w, r):
    """Create a circle (center_x, center_y, radius) that encompases a given
    rectangle (x, y, l, w, r)"""
    return (x, y, math.sqrt((l/2) ** 2 + (w/2) ** 2))

def circle_coverage(x, y, l, w, r, subdivision_count=3):
    """Create subdivision_count^2 number of circles (center_x, center_y, radius)
    that together encompass a given rectangle (x, y, l, w, r) with r in
    degrees indicating counterclockwise rotation"""

    circles = []
    for subrectangle in get_subrectangles(x, y, l, w, r, subdivision_count):
        circles.append(single_circle_coverage(*subrectangle))

    return circles

def random_nonintersecting_circle(center_x, center_y, upper_b, lower_b, left_b, right_b, other_circles):
    """Given other_circles (as a list of (cx, cy, radius)), find the largest circle
       that does not intersect with other circles

        It takes the following parameters:
        center_x, center_y: Coordinates of the center of the new circle.
        upper_b, lower_b, left_b, right_b: Boundaries of the region where the circle can be placed.
        other_circles: A list of tuples representing existing circles (each tuple contains cx, cy, and radius).
    
    """

    # radius is initialized to the maximum possible floating-point value
    radius = sys.float_info.max

    # Iterates through each existing circle
    for (other_x, other_y, other_radius) in other_circles:

        # Calculates the distance between the center of the new circle and 
        #  the center of the existing circle using the Euclidean distance formula.
        distance = math.sqrt((other_x - center_x)**2 + (other_y - center_y)**2)


        # Calculates the distance to the nearest boundary
        boundary_distance = get_boundary_distance(center_x, center_y, upper_b, lower_b, left_b, right_b)
        
        # To ensure that the new circle doesn't overlap with any existing circle or exceed the boundaries
        radius = min([radius, distance - other_radius, boundary_distance])

    if radius <= 0:
        # No non-intersecting circle can be placed
        return None
    else:
        coeff = random.uniform(0.5, 0.9)
        return center_x, center_y, coeff * radius

def random_nonintersecting_rectangle(center_x, center_y, upper_b, lower_b, left_b, right_b, other_rectangles, subdivision_count=3):
    """Given other_rectangles (as a list of (x, y, l, w, r)), return a random rectangle
    inside the largest circle that does not intersect with the circles that cover the other rectangles."""

    all_other_circles = []
    for other_rectangle in other_rectangles:
        all_other_circles += circle_coverage(*other_rectangle, subdivision_count)
    circle = random_nonintersecting_circle(center_x, center_y, upper_b, lower_b, left_b, right_b, all_other_circles)
    if circle is not None:
        return random_rectangle(*circle)
    else:
        return None

def get_boundary_distance(center_x, center_y, upper_b, lower_b, left_b, right_b):
    """
        function takes six arguments:
            center_x: The x-coordinate of the center point.
            center_y: The y-coordinate of the center point.
            upper_b: The y-coordinate of the upper boundary.
            lower_b: The y-coordinate of the lower boundary.
            left_b: The x-coordinate of the left boundary.
            right_b: The x-coordinate of the right boundary.
    """

    # Calculate the distances between the center point and each of the boundaries
    upper_distance = upper_b - center_y
    lower_distance = center_y - lower_b
    left_distance = center_x - left_b
    right_distance = right_b - center_x

    # Calculate the shortest distance between the center point and any of the boundaries
    return min([upper_distance, lower_distance, left_distance, right_distance])

def plot_rectangle(rectangles):
    for (x, y, l, w, r) in rectangles:
        pl.plot([x], [y], color="#FF000030", marker='o', markersize=10)
        rect = patches.Rectangle((x - l / 2, y - w / 2), l, w, color="#FF000030", alpha=0.10, angle=r, rotation_point='center')
        ax = pl.gca()
        ax.add_patch(rect)
    pl.figure()
    pl.plot()
    pl.xlim([-10, 150])
    pl.ylim([-10, 150])
    pl.show()



if __name__ == "__main__":


    def plot_rectangle_and_coverage_circles(rectangle, color, draw_coverage=True, subdivision_count=3):
        (x, y, l, w, r) = rectangle
        pl.plot([x], [y], color=color, marker='o', markersize=10)
        rect = patches.Rectangle((x-l/2,y-w/2), l, w, color=color, alpha=0.10, angle=r, rotation_point='center')
        ax = pl.gca()

        ax.add_patch(rect)

        if draw_coverage:
            circles = circle_coverage(x, y, l, w, r, subdivision_count)
            for (cx, cy, radius) in circles:
                pl.plot([cx], [cy], marker="x", markersize=5, color="red")
                circle = patches.Circle((cx, cy), radius, color="#FF000030", alpha=0.1)
                ax.add_patch(circle)

    pl.figure()
    pl.plot()
    pl.xlim([-10, 150])
    pl.ylim([-10, 150])

    subdivision_count = 3
    # x, y, l, w, r
    rectangle1 = (10, 30, 40, 40, 10)
    plot_rectangle_and_coverage_circles(rectangle1, "blue", subdivision_count=subdivision_count)

    rectangle2 = (90, 30, 50, 50, 90)
    plot_rectangle_and_coverage_circles(rectangle2, "blue", subdivision_count=subdivision_count)

    new_rectangle = random_nonintersecting_rectangle(35, 15, 100, -100, -100, 100, [rectangle1, rectangle2], subdivision_count=subdivision_count)
    plot_rectangle_and_coverage_circles(new_rectangle, "green", True, 1)

    pl.show()