import numpy as np
from .dashed_line import DashedLine

class YellowDashedLine(DashedLine):
  def draw(self, image: np.ndarray, thickness: int = 2, dash_length: int = 10):
    """
    Draw a yellow line on an image.

    Parameters:
        image (numpy.ndarray): The image on which to draw the line.
        thickness (int): Thickness of the line.
    """
    super().draw(image, color=(0, 255, 255), thickness=thickness, dash_length=dash_length)