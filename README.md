# circuit-detection

## Installation

Install dependencies (Python 3.9+ recommended):

```bash
pip install -r requirements.txt
```

## Model files

- `.pickle` config files included in (`models/mechanic/circuit.pickle`) and (`models/electron/symbol.pickle`)
- `.pth` weight files are not committed (too large), it will auto downloaded from release when first time using

### download manually

you can downloa pth files manually from [GitHub Releases](https://github.com/CHENMENG-HAN/circuit-detection/releases)\
and place the files in (`models/mechanic`) or (`models/electron`)

## Usage

```python
from circuit_detection import detect_circuit

results = detect_circuit(doc)

```