from __future__ import annotations
import cv2
from ..point import Point
from pydantic import BaseModel
import numpy as np

class Line(BaseModel):
  start: Point
  end: Point

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
      tuple: ((x1, y1), (x2, y2)) representing the start and end points.
    """
    return self.start.to_XY_tuple(), self.end.to_XY_tuple()

  def draw(self, image: np.ndarray, color: tuple[int, int, int] = (255, 255, 0), thickness: int = 2):
    """
    Draw the line across the entire image using OpenCV.

    Parameters:
      image (numpy.ndarray): The image on which to draw the line.
      color (tuple): Color of the line in BGR format.
      thickness (int): Thickness of the line.
    """
    height, width = image.shape[:2]
    x1, y1 = self.start.to_XY_tuple()
    x2, y2 = self.end.to_XY_tuple()
    
    # Calculate the slope
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = float('inf')  # Vertical line
    
    # Extend line to image boundaries
    if abs(slope) == float('inf'):  # Vertical line
        extended_start = (x1, 0)
        extended_end = (x1, height - 1)
    else:
        # Line equations to find intersections
        y_at_x0 = int(y1 + slope * (0 - x1))  # Intersection with left boundary (x = 0)
        y_at_x_max = int(y1 + slope * (width - 1 - x1))  # Intersection with right boundary (x = width - 1)
        x_at_y0 = int(x1 + (0 - y1) / slope)  # Intersection with top boundary (y = 0)
        x_at_y_max = int(x1 + (height - 1 - y1) / slope)  # Intersection with bottom boundary (y = height - 1)

        # Determine points within image bounds
        points = [
            (0, y_at_x0), (width - 1, y_at_x_max),  # Horizontal boundaries
            (x_at_y0, 0), (x_at_y_max, height - 1)  # Vertical boundaries
        ]
        valid_points = [(x, y) for x, y in points if 0 <= x < width and 0 <= y < height]

        if len(valid_points) == 2:
            extended_start, extended_end = valid_points
        else:
            return  # Line does not intersect the image

    # Draw the extended line
    cv2.line(image, extended_start, extended_end, color, thickness)

  def project(self, projection_matrix: np.ndarray) -> Line:
    """
    Project a 3D line to 2D using a projection matrix.

    Parameters:
      projection_matrix (numpy.ndarray): 3x4 combined projection matrix.

    Returns:
      Line: A new Line object with projected 2D points.
    """
    if self.start.z is None or self.end.z is None:
      raise ValueError("Cannot project a line with 2D points; both points must be 3D.")
    projected_start = self.start.project(projection_matrix)
    projected_end = self.end.project(projection_matrix)
    return self.__class__(start=projected_start, end=projected_end)