import random
from typing import Type
from .road_segment import RoadSegment
from ..point import Point
from ..lines import *

class RoadSegmentGenerator:
  MIN_LANE_NUMBER = 1
  MAX_LANE_NUMBER = 10

  @classmethod
  def generate(
    cls, 
    lane_number: int = None,
    forward_lane_number: int = None,
    lane_width: int = 10,
    lane_length: int = 800,
    offset: int = 5,
    road_center: Point = Point(x = 500, y = 100, z = 0),
    seed: int = None
  ) -> RoadSegment:
    random.seed(seed)

    lane_number = lane_number if lane_number is not None else random.randint(cls.MIN_LANE_NUMBER, cls.MAX_LANE_NUMBER)
    forward_lane_number = forward_lane_number if forward_lane_number is not None else random.randint(0, lane_number)

    if forward_lane_number > lane_number:
      raise RuntimeError("Forward line number ({}) cannot exceed total line number ({})".format(forward_lane_number, lane_number))
    
    if lane_number < cls.MIN_LANE_NUMBER:
      raise RuntimeError("Lane number ({}) can't be lower than MIN_LANE_NUMBER ({})".format(lane_number, cls.MIN_LANE_NUMBER))

    if lane_number > cls.MAX_LANE_NUMBER:
      raise RuntimeError("Lane number ({}) can't be higher than MIN_LANE_NUMBER ({})".format(lane_number, cls.MAX_LANE_NUMBER))
    
    road = RoadSegment()

    for i in range(lane_number + 1):
      offset_x = (i if i <= forward_lane_number else i - lane_number - 1) * lane_width

      start = Point.from_source_offset(
        source=road_center,
        offset=Point(x = offset_x, y = 0, z = 0)
      )
      end = Point.from_source_offset(
        source=road_center, 
        offset=Point(x = offset_x, y = lane_length, z = 0)
      )
      
      line_class: Type[Line]

      is_middle_line = i == 0 and lane_number > 1
      is_edge_line = i == lane_number or i == forward_lane_number - 1

      if is_middle_line:
        line_class = random.choice([SolidLine, DoubleLine, DoubleDashedLine, MixedDoubleLine])
      elif is_edge_line:
        line_class = random.choice([Line, SolidLine, YellowLine, YellowDashedLine])
      else:
        line_class = random.choice([DashedLine])

      line = line_class(start=start, end=end)

      road.add_line(line)

    return road