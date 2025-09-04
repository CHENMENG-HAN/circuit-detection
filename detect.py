from utils import setup_predictor
import fitz

def detect_circuit(pdf_doc: fitz.Document, threshold = 0.7):
    predictor = setup_predictor(weight_path = "models/mechanic/circuit.pth", config_path = "models/mechanic/circuit.pickle", threshold = threshold)