import numpy as np
from .line import Line

class SolidLine(Line):
  def draw(self, image: np.ndarray, thickness: int = 2):
    """
    Draw a yellow line on an image.

    Parameters:
      image (numpy.ndarray): The image on which to draw the line.
      thickness (int): Thickness of the line.
      dash_length (int): Length of each dash.
    """
    color=(255, 255, 255)
    super().draw(image, color=color, thickness=thickness)