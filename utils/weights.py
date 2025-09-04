from pathlib import Path
import urllib.request

GITHUB_OWNER = "CHENMENG-HAN"
GITHUB_REPO  = "circuit-detection"

MODEL_INFO = {
    "mechanic": {
        "weight": "models/mechanic/circuit.pth",
        "config": "models/mechanic/circuit.pickle",
        "tag": "v0.1",
        "asset": "circuit.pth",
    },
    "electron": {
        "weight": "models/electron/symbol.pth",
        "config": "models/electron/symbol.pickle",
        "tag": "v0.1",
        "asset": "symbol.pth",
    }
}

PATH_ROOT = Path(__file__).resolve().parents[1]

def _redirect_path(path: str) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path
    
    return (PATH_ROOT / path).resolve()

def _download(url: str, destination: Path):
    destination.parent.mkdir(parents = True, exist_ok = True)
    print(f"downloading {url} -> {destination}, do not stop the process")
    urllib.request.urlretrieve(url, destination)

def ensure_weight(model = "mechanic"):
    if model not in MODEL_INFO:
        raise ValueError("model must be mechanic or electron")
    
    info = MODEL_INFO[model]
    weight_path = _redirect_path(info["weight"])
    config_path = _redirect_path(info["config"])

    if not weight_path.exists():
        url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{info['tag']}/{info['asset']}"
        _download(url, weight_path)

    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    
    return str(weight_path), str(config_path)