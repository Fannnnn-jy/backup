from .random import Random
from .exploration import Exploration
from .confidence import Confidence


def get_planner(cfg, device):
    planner_cfg = cfg.planner
    planner_up_axis = str(cfg.scene.get("planner_up_axis", "z")).lower()
    if planner_cfg.type == "random":
        return Random(planner_cfg, device, planner_up_axis=planner_up_axis)
    elif planner_cfg.type == "exploration":
        return Exploration(planner_cfg, device, planner_up_axis=planner_up_axis)
    elif planner_cfg.type == "confidence":
        return Confidence(planner_cfg, device, planner_up_axis=planner_up_axis)
    else:
        raise NotImplementedError
