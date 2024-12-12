import random
from typing import Type, Optional
from .road_segment import RoadSegment
from ..point import Point
from ..lines import *

class RoadSegmentGenerator:
  MIN_LANE_NUMBER = 1
  MAX_LANE_NUMBER = 6

  @classmethod
  def generate(
    cls, 
    lane_number: Optional[int] = None,
    forward_lane_number: Optional[int] = None,
    lane_width: int = 10,
    lane_length: int = 800,
    offset: int = 5,
    road_center: Point = Point(x=0, y=500, z=0),
    seed: Optional[int] = None
  ) -> RoadSegment:
    """
    Generate a road segment with configurable parameters.

    Parameters:
      lane_number (int, optional): Total number of lanes. 
        If None, randomly selected between MIN_LANE_NUMBER and MAX_LANE_NUMBER.
      forward_lane_number (int, optional): Number of forward lanes. 
        If None, randomly selected up to total lane number.
      lane_width (int): Width of each lane.
      lane_length (int): Length of the road segment.
      offset (int): Offset between lanes.
      road_center (Point): Center point of the road.
      seed (int, optional): Random seed for reproducibility.

    Returns:
      RoadSegment: Generated road segment with lines.

    Raises:
      ValueError: If lane configuration is invalid.
    """
    # Set random seed for reproducibility
    if seed is not None:
      random.seed(seed)

    # Validate and set lane numbers
    try:
      lane_number = (
        lane_number if lane_number is not None 
        else random.randint(cls.MIN_LANE_NUMBER, cls.MAX_LANE_NUMBER)
      )
      
      forward_lane_number = (
        forward_lane_number if forward_lane_number is not None 
        else int(lane_number // 2)
        # else random.randint(0, lane_number)
      )

      # Comprehensive validation
      if not cls.MIN_LANE_NUMBER <= lane_number <= cls.MAX_LANE_NUMBER:
        raise ValueError(
          f"Lane number must be between {cls.MIN_LANE_NUMBER} "
          f"and {cls.MAX_LANE_NUMBER}. Got {lane_number}."
        )

      if forward_lane_number > lane_number:
        raise ValueError(
          f"Forward lane number ({forward_lane_number}) "
          f"cannot exceed total lane number ({lane_number})."
        )

    except ValueError as e:
      raise ValueError(f"Invalid lane configuration: {e}")

    # Initialize road segment
    road = RoadSegment()

    # Line type selection strategies
    def select_middle_line_class():
      return random.choice([
        SolidLine, 
        DoubleLine, 
        DoubleDashedLine, 
        MixedDoubleLine
      ])

    def select_edge_line_class():
      return random.choice([
        Line, 
        SolidLine, 
        YellowLine, 
        YellowDashedLine
      ])

    def select_lane_line_class():
      return random.choice([DashedLine])

    # Generate lines for the road segment
    for i in range(lane_number + 1):
      # Calculate lane offset
      offset_x = i * lane_width

      # Create start and end points
      start = Point.from_source_offset(
        source=road_center, 
        offset=Point(x=offset_x, y=0, z=0)
      )
      end = Point.from_source_offset(
        source=road_center, 
        offset=Point(x=offset_x, y=lane_length, z=0)
      )
      
      # Select line class based on lane position
      if i == forward_lane_number and lane_number > 1:
        line_class = select_middle_line_class()
      elif i == 0 or i == lane_number:
        line_class = select_edge_line_class()
      else:
        line_class = select_lane_line_class()

      # Create and add line to road segment
      line = line_class(start=start, end=end)
      road.add_line(line)

    return road