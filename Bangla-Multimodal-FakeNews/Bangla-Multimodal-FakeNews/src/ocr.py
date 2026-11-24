"""EasyOCR (Bengali + English): extract text from meme images."""


def load_ocr_reader(langs, use_gpu: bool):
    import easyocr

    return easyocr.Reader(langs, gpu=use_gpu)


def ocr_text(reader, image_path: str) -> str:
    return " ".join(reader.readtext(str(image_path), detail=0, paragraph=True)).strip()


def ocr_pil(reader, pil_image) -> str:
    """Same reading as ocr_text(), for an image already loaded in memory (the demo app upload)."""
    import numpy as np

    arr = np.array(pil_image.convert("RGB"))
    return " ".join(reader.readtext(arr, detail=0, paragraph=True)).strip()
