import numpy as np

class Point:
    def __init__(self, x: int, y: int):
        """
        Initialize a point with x and y coordinates.

        Parameters:
            x (int): X-coordinate of the point.
            y (int): Y-coordinate of the point.
        """
        self.x = x
        self.y = y

    def to_tuple(self):
        """
        Convert the point to a tuple.

        Returns:
            tuple: (x, y) coordinates of the point.
        """
        return self.x, self.y

    def distance_to(self, other_point):
        """
        Calculate the Euclidean distance to another point.

        Parameters:
            other_point (Point): The other point to calculate distance to.

        Returns:
            float: Euclidean distance.
        """
        return np.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"