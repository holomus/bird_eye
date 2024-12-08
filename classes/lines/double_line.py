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
        Draw a double line on an image.

        Parameters:
            image (numpy.ndarray): The image on which to draw the line.
            color (tuple): Color of the line in BGR format.
            thickness (int): Thickness of the lines.
            offset (int): Offset between the two lines.
        """
        # Draw the main line
        cv2.line(image, self.start.to_XY_tuple(), self.end.to_XY_tuple(), color, thickness)

        # Compute offset points and draw the second line
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = (dx**2 + dy**2)**0.5
        offset_x = -offset * dy / length
        offset_y = offset * dx / length

        start_offset = (int(self.start.x + offset_x), int(self.start.y + offset_y))
        end_offset = (int(self.end.x + offset_x), int(self.end.y + offset_y))

        cv2.line(image, start_offset, end_offset, color, thickness)
