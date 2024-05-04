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

    # try:
    font_substrings, bbox_substrings = extract_string(image)
    """
        BBOX SUBSTRING IS A LIST CONTAINING THE FOLLOWING

        - ([LIST OF COORDINATES -> X, Y, W, H]) FOR EACH SUBSTRING "HELLO WORLD"
            WHICH OCCURS AN HORIZONTAL SECTION
        - A TUPLE INDICATING THE START AND END Y COORDS FOR THE HORIZONTAL STRIP UNDER CONSIDERATION

        EFFECTIVELY, OUR BBOX COORDINATE FINAL IN THE ORIG IMAGE WILL BE
        FOR ONE ROW, ASSUME THERE IS JUST ONE HELLO WORLD,  
        IMAGE[X: X+W, START:START+END]
    """
    

    output = classify_fonts(font_substrings)

    count = 0
    for i in range(len(bbox_substrings)):
        for index, x_pos in enumerate(bbox_substrings[i][:-1]):
            x1 = x_pos[0]
            x2 = x_pos[0] + x_pos[2]
            y1 = bbox_substrings[i][-1][0]
            y2 = bbox_substrings[i][-1][1]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, output[count], (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            count += 1

            
    cv2.imwrite(f"assets/image_bbox_final.png", image)

    return output
    # except Exception as e:
    #     logger.error(f"Couldnt run classification of extracted font, {e}")

    

if __name__ == '__main__':
    output = run_custom_font_detector("image2.png")
    print(output)
    logger.info(output)
