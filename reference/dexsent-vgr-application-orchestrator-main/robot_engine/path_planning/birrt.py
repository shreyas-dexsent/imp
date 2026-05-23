from __future__ import annotations

from robot_engine.path_planning.rrt_connect import RRTConnectPlanner


class BiRRTPlanner(RRTConnectPlanner):
    planner_name = "BIRRT"

