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

  def draw(self, image: np.ndarray, color: tuple[int, int, int] = (255, 255, 0), thickness: int = 1):
    """
    Draw the line on a given image using OpenCV.

    Parameters:
      image (numpy.ndarray): The image on which to draw the line.
      color (tuple): Color of the line in BGR format.
      thickness (int): Thickness of the line.
    """
    cv2.line(image, self.start.to_XY_tuple(), self.end.to_XY_tuple(), color, thickness)

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