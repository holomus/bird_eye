import numpy as np
from .line import Line

class YellowLine(Line):
  def draw(self, image: np.ndarray, thickness: int = 2):
    """
    Draw a yellow line on an image.

    Parameters:
        image (numpy.ndarray): The image on which to draw the line.
        thickness (int): Thickness of the line.
    """
    color = (0, 255, 255)  # Yellow in BGR
    super().draw(image, color=color, thickness=thickness)