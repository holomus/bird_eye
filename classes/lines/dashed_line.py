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
    Draw a dashed line across the entire image.

    Parameters:
      image (numpy.ndarray): The image on which to draw the line.
      color (tuple): Color of the line in BGR format.
      thickness (int): Thickness of the line.
      dash_length (int): Length of each dash.
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

    # Compute the full-length dashed line
    x1, y1 = extended_start
    x2, y2 = extended_end
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # Length of the line
    num_dashes = int(length // (2 * dash_length))

    for i in range(num_dashes):
      # Start and end points for each dash
      start_dash = (
        int(x1 + (2 * i) / num_dashes * (x2 - x1)),
        int(y1 + (2 * i) / num_dashes * (y2 - y1)),
      )
      end_dash = (
        int(x1 + (2 * i + 1) / num_dashes * (x2 - x1)),
        int(y1 + (2 * i + 1) / num_dashes * (y2 - y1)),
      )
      
      # Draw the dash
      cv2.line(image, start_dash, end_dash, color, thickness)

