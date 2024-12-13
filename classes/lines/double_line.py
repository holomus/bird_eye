import cv2
import numpy as np
from .line import Line

class DoubleLine(Line):
  def draw(
    self, 
    image: np.ndarray, 
    color: tuple[int, int, int] = (255, 255, 255), 
    thickness: int = 2, 
    offset: int = 5
  ):
    """
    Draw a double line across the entire image.

    Parameters:
      image (numpy.ndarray): The image on which to draw the lines.
      color (tuple): Color of the lines in BGR format.
      thickness (int): Thickness of the lines.
      offset (int): Offset between the two lines.
    """
    height, width = image.shape[:2]
    x1, y1 = self.start.to_XY_tuple()
    x2, y2 = self.end.to_XY_tuple()

    # Calculate the slope
    if x2 != x1:
      slope = (y2 - y1) / (x2 - x1)
    else:
      slope = float('inf')  # Vertical line

    # Extend the main line to image boundaries
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

    # Draw the main line
    x1, y1 = extended_start
    x2, y2 = extended_end
    cv2.line(image, extended_start, extended_end, color, thickness)

    # Compute offset points for the second line
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    offset_x = -offset * dy / length
    offset_y = offset * dx / length

    start_offset = (int(x1 + offset_x), int(y1 + offset_y))
    end_offset = (int(x2 + offset_x), int(y2 + offset_y))

    # Draw the second line
    cv2.line(image, start_offset, end_offset, color, thickness)
