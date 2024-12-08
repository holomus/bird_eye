import numpy as np
from .line import Line
from .dashed_line import DashedLine
from ..point import Point

class DoubleDashedLine(Line):
  def draw(
      self, 
      image: np.ndarray, 
      color: tuple[int, int, int] = (255, 255, 255), 
      thickness: int = 2, 
      dash_length: int = 10, 
      offset: int = 5
    ):
    """
    Draw two parallel dashed lines on the image.

    Parameters:
      image (numpy.ndarray): The image on which to draw the lines.
      color (tuple): Color of the dashed lines in BGR format.
      thickness (int): Thickness of the lines.
      dash_length (int): Length of each dash.
      offset (int): Offset between the two lines.
    """
    # Draw the first dashed line (original position)
    super().draw(image, color=color, thickness=thickness, dash_length=dash_length)
    
    # Calculate offset points
    dx = self.end.x - self.start.x
    dy = self.end.y - self.start.y
    length = (dx**2 + dy**2)**0.5
    offset_x = -offset * dy / length
    offset_y = offset * dx / length

    # Offset start and end points for the second dashed line
    start_offset = Point(self.start.x + offset_x, self.start.y + offset_y)
    end_offset = Point(self.end.x + offset_x, self.end.y + offset_y)
    
    dashed_offset_line = DashedLine(start_offset, end_offset)
    dashed_offset_line.draw(image, color=color, thickness=thickness, dash_length=dash_length)
