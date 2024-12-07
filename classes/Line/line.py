import cv2
from ..point import Point

class Line:
    def __init__(self, start: Point, end: Point):
        """
        Initialize a line with a start and end point.

        Parameters:
            start_point (Point): The starting point of the line.
            end_point (Point): The ending point of the line.
        """
        self.start = start
        self.end = end

    def length(self):
        """
        Calculate the length of the line.

        Returns:
            float: Length of the line (Euclidean distance between start and end points).
        """
        return self.start.distance_to(self.end)

    def to_tuple(self):
        """
        Convert the line to a tuple of point tuples.

        Returns:
            tuple: ((x1, y1, z1), (x2, y2, z2)) representing the start and end points.
        """
        return self.start.to_tuple(), self.end.to_tuple()

    def draw(self, image, color=(255, 255, 255), thickness=2):
        """
        Draw the line on a given image using OpenCV.

        Parameters:
            image (numpy.ndarray): The image on which to draw the line.
            color (tuple): Color of the line in BGR format.
            thickness (int): Thickness of the line.
        """
        cv2.line(image, self.start.to_tuple(), self.end.to_tuple(), color, thickness)

    def __repr__(self):
        return f"Line(start_point={self.start}, end_point={self.end})"