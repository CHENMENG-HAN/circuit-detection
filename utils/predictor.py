from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from pathlib import Path
import pickle

PATH_ROOT = Path(__file__).resolve().parents[1]

def _redirect_path(path: str) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path
    
    return (PATH_ROOT / path).resolve()

def setup_predictor(weight_path: str, config_path: str, threshold):
    weight_path = _redirect_path(weight_path)
    config_path = _redirect_path(config_path)

    if not weight_path.exists():
        raise FileNotFoundError(f"{weight_path} not found")
    
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    with open(str(config_path), "rb") as f:
        cfg = pickle.load(f)
    
    cfg.MODEL.WEIGHTS = str(weight_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda" # or cpu

    predictor = DefaultPredictor(cfg)

    return predictor