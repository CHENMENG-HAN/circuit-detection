from .utils import setup_predictor, pixmap_to_bgr
import fitz

class PDFPatternLocation:
    def __init__(self, page_idx, x0: float, y0: float, x1: float, y1: float):
        self.page_idx = page_idx
        # normalized coord
        self.x0 = min(x0, x1)
        self.y0 = min(y0, y1)
        self.x1 = max(x0, x1)
        self.y1 = max(y0, y1)

MODEL_PATHS = {
    "mechanic": {
        "weight": "models/mechanic/circuit.pth",
        "config": "models/mechanic/circuit.pickle",
    },
    "electron": {
        "weight": "models/electron/symbol.pth",
        "config": "models/electron/symbol.pickle",
    }
}

def detect_circuit(pdf_doc: fitz.Document, threshold = 0.7, dpi = 300, model = "mechanic"):
    if model not in MODEL_PATHS:
        raise ValueError("model must be mechanic or electron")

    weight_path = MODEL_PATHS[model]["weight"]
    config_path = MODEL_PATHS[model]["config"]

    predictor = setup_predictor(weight_path = weight_path, config_path = config_path, threshold = threshold)
    
    results: list[PDFPatternLocation] = [] # store all the predict result

    for i in range(pdf_doc.page_count):
        page = pdf_doc.load_page(i)
        pix = page.get_pixmap(dpi = dpi, colorspace = fitz.csRGB, alpha = False)
        image = pixmap_to_bgr(pix)
        
        output = predictor(image)
        instances = output["instances"].to("cpu")

        # no circuit founded in page
        if len(instances) == 0: 
            continue
        
        height, width = image.shape[:2] # page's orignal height and width

        for box in instances.pred_boxes.tensor:
            x0, y0, x1, y1 = box.tolist()
            nor_x0, nor_y0, nor_x1, nor_y1 = x0 / width, y0 / height, x1 / width, y1 / height

            results.append(PDFPatternLocation(page_idx = i, x0 = nor_x0, y0 = nor_y0, x1 = nor_x1, y1 = nor_y1))

    return results
