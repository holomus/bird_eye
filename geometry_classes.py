import numpy as np


class Point:
    def __init__(self, x: np.uint64, y: np.uint64) -> None:
        self.coor = np.array([x, y])

    def __call__(self):
        return self.coor


class Line:
    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end
