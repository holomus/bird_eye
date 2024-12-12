import cv2
import numpy as np
from .line import Line

class DashedLine(Line):
  def _is_further_on_line(self, point):
    """
    Check if a point is further along a line segment.

    Parameters:
      point (tuple): The point to check (x, y).
    
    Returns:
      bool: True if the point is further along the line, False otherwise.
    """
    x1, y1 = self.start.to_XY_tuple()
    x2, y2 = self.end.to_XY_tuple()
    px, py = point

    # Check for collinearity
    if (px - x1) * (y2 - y1) != (py - y1) * (x2 - x1):
      return False  # Not on the same line

    # Calculate t parameter
    t_x = (px - x1) / (x2 - x1) if x2 != x1 else float('inf')
    t_y = (py - y1) / (y2 - y1) if y2 != y1 else float('inf')

    # Compare t_x and t_y for consistency
    t = t_x if x2 != x1 else t_y  # Use valid t

    return t > 1

  def draw(
    self, 
    image: np.ndarray, 
    color: tuple[int, int, int] = (255, 255, 255), 
    thickness: int = 2, 
    dash_length: int = 10
  ):
    """
    Draw a dashed line on an image.

    Parameters:
      image (numpy.ndarray): The image on which to draw the line.
      color (tuple): Color of the line in BGR format.
      thickness (int): Thickness of the line.
      dash_length (int): Length of each dash.
    """
    num_dashes = int(self.length() // (2 * dash_length))
    for i in range(num_dashes):
      start_dash = (
        int(self.start.x + (2 * i) / num_dashes * (self.end.x - self.start.x)),
        int(self.start.y + (2 * i) / num_dashes * (self.end.y - self.start.y)),
      )
      end_dash = (
        int(self.start.x + (2 * i + 1) / num_dashes * (self.end.x - self.start.x)),
        int(self.start.y + (2 * i + 1) / num_dashes * (self.end.y - self.start.y)),
      )
      
      if self._is_further_on_line(end_dash):
        break
      cv2.line(image, start_dash, end_dash, color, thickness)
