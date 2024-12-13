from __future__ import annotations
import numpy as np
from ..lines import Line
from ..point import Point
from pydantic import BaseModel
import random

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
      line.draw(image=image, thickness=thickness)

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

  def random_adjacent_middle_point(self) -> Point:
    """
    Randomly select two adjacent lines and return their middle point along the x-axis.

    Returns:
      Point: The middle point (x, y, z) along the x-axis of the two adjacent lines.
    """
    if len(self.lines) < 2:
      raise ValueError("Not enough lines to select two adjacent ones.")
    
    # Randomly select an index for the first line
    index = random.randint(0, len(self.lines) - 2)
    
    # Get the two adjacent lines
    line1 = self.lines[index]
    line2 = self.lines[index + 1]
    
    # Calculate the middle point along the x-axis
    middle_x1 = (line1.start.x + line1.end.x) / 2
    middle_x2 = (line2.start.x + line2.end.x) / 2
    middle_x = int((middle_x1 + middle_x2) // 2)

    # Average the y-coordinates for a central point
    middle_y1 = (line1.start.y + line1.end.y) / 2
    middle_y2 = (line2.start.y + line2.end.y) / 2
    middle_y = int((middle_y1 + middle_y2) // 2)

    # Average the y-coordinates for a central point
    middle_z1 = (line1.start.z + line1.end.z) / 2
    middle_z2 = (line2.start.z + line2.end.z) / 2
    middle_z = int((middle_z1 + middle_z2) // 2)

    return Point(x=middle_x, y=middle_y, z=middle_z)