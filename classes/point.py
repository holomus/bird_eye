from __future__ import annotations
import numpy as np
from pydantic import BaseModel

class Point(BaseModel):
  x: float
  y: float
  z: float | None = None

  @classmethod
  def from_source_offset(cls, source: Point, offset: Point) -> Point:
    """
    Create a new point by applying an offset to a source point.

    Parameters:
      source (Point): The reference point.
      offset (Point): The offset to apply.

    Returns:
      Point: The resulting point.
    """
    if source.z is None and offset.z is not None:
      raise ValueError("Cannot add a 3D offset to a 2D point.")
    z = None if source.z is None else source.z + (offset.z or 0)
    return cls(x = source.x + offset.x, y = source.y + offset.y, z = z)

  def to_XY_tuple(self):
    """
    Convert the point to a tuple.

    Returns:
      tuple: (x, y) coordinates of the point.
    """
    return int(self.x), int(self.y)

  def distance_to(self, other: Point):
    """
    Calculate the Euclidean distance to another point.

    Parameters:
      other (Point): The other point to calculate distance to.

    Returns:
      float: Euclidean distance.
    """
    if self.z is None and other.z is None:
      return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    if self.z is None or other.z is None:
      raise ValueError("Cannot calculate distance between 2D and 3D points.")
    return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)
  
  def project(self, projection_matrix: np.ndarray):
    """
    Project a 3D point to 2D using a full projection matrix (intrinsic + extrinsic).

    Parameters:
      projection_matrix (numpy.ndarray): 3x4 combined projection matrix.

    Returns:
      Point: A 2D point representing the projected coordinates.
    """
    if self.z is None:
      raise ValueError("Cannot project a 2D point; only 3D points can be projected.")

    # Convert Point to homogeneous coordinates
    point_homogeneous = np.array([self.x, self.y, self.z, 1])
    # Apply the projection matrix
    projected = projection_matrix @ point_homogeneous
    # Normalize by the third coordinate to get (x, y)
    x, y, w = projected[:3]
    if w == 0:
      raise ValueError("Homogeneous coordinate w cannot be zero.")
    return Point(x= x / w, y = y / w)