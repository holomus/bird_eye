import cv2
import numpy as np
from .line import Line

class DashedLine(Line):
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
            cv2.line(image, start_dash, end_dash, color, thickness)
