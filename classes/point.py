import numpy as np
from __future__ import annotations

class Point:
    def __init__(self, x: float, y: float, z: float):
        """
        Initialize a point with x, y and z coordinates.

        Parameters:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.
            z (float): Z-coordinate of the point.
        """
        self.x = x
        self.y = y
        self.z = z

    def to_tuple(self):
        """
        Convert the point to a tuple.

        Returns:
            tuple: (x, y, z) coordinates of the point.
        """
        return self.x, self.y, self.z

    def distance_to(self, other: Point):
        """
        Calculate the Euclidean distance to another point.

        Parameters:
            other (Point): The other point to calculate distance to.

        Returns:
            float: Euclidean distance.
        """
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"