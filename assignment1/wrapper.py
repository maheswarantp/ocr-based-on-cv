from cv.extract_string import extract_string
from inference.classify_font import classify_fonts
from typing import List, Dict
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

def run_custom_font_detector(image_path: str):
    image = cv2.imread(image_path)
    logger.info("Read image")

    try:
        font_substrings = extract_string(image)   
        output = classify_fonts(font_substrings)
        return output
    except Exception as e:
        logger.error(f"Couldnt run classification of extracted font, {e}")

    

if __name__ == '__main__':
    output = run_custom_font_detector("image.png")
    print(output)
    logger.info(output)
