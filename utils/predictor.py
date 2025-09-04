from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import pickle

def setup_predictor(weight_path: str, config_path: str, threshold):
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda" # or cpu

    predictor = DefaultPredictor(cfg)

    return predictor