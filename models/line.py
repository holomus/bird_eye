import cv2
from .point import Point

class Line:
    def __init__(self, start_point: Point, end_point: Point):
        """
        Initialize a line with a start and end point.

        Parameters:
            start_point (Point): The starting point of the line.
            end_point (Point): The ending point of the line.
        """
        self.start_point = start_point
        self.end_point = end_point

    def length(self):
        """
        Calculate the length of the line.

        Returns:
            float: Length of the line (Euclidean distance between start and end points).
        """
        return self.start_point.distance_to(self.end_point)

    def to_tuple(self):
        """
        Convert the line to a tuple of point tuples.

        Returns:
            tuple: ((x1, y1), (x2, y2)) representing the start and end points.
        """
        return self.start_point.to_tuple(), self.end_point.to_tuple()

    def draw(self, image, color=(255, 255, 255), thickness=2):
        """
        Draw the line on a given image using OpenCV.

        Parameters:
            image (numpy.ndarray): The image on which to draw the line.
            color (tuple): Color of the line in BGR format.
            thickness (int): Thickness of the line.
        """
        cv2.line(image, self.start_point.to_tuple(), self.end_point.to_tuple(), color, thickness)

    def __repr__(self):
        return f"Line(start_point={self.start_point}, end_point={self.end_point})"