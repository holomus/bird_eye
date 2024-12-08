from __future__ import annotations
import numpy as np
from ..lines import Line
from pydantic import BaseModel

class RoadSegment(BaseModel):
  lines: list[Line] = []

  def add_line(self, line: Line):
    """
    Add a line to the road segment.

    Parameters:
      line (Line): A Line object (or subclass) to add to the road segment.
    """
    self.lines.append(line)

  def draw(self, image: np.ndarray, thickness: int = 2):
    """
    Draw all lines in the road segment on the image.

    Parameters:
      image (numpy.ndarray): The image on which to draw the road segment.
    """
    for line in self.lines:
      line.draw(image, thickness)

  def project(self, projection_matrix: np.ndarray) -> RoadSegment:
    """
    Project all lines in the road segment from 3D to 2D.

    Parameters:
      projection_matrix (numpy.ndarray): 3x4 combined projection matrix.

    Returns:
      RoadSegment: A new RoadSegment with projected 2D lines.
    """
    projected_segment = RoadSegment()
    for line in self.lines:
      if line.start.z is None or line.end.z is None:
        raise ValueError("Cannot project a road segment containing 2D lines.")
      projected_segment.add_line(line.project(projection_matrix))
    return projected_segment
