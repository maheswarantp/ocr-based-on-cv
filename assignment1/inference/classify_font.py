import numpy as np
from typing import List
from training_scripts.train import init_datagen
import cv2
from model.cnn_model import return_model
import logging
import uuid

logger = logging.getLogger(__name__)

model = return_model((256, 256), 8)

labels = ['dancingscript',
 'fredoka',
 'notosans',
 'opensans',
 'oswald',
 'ptserif',
 'roboto',
 'ubuntu']


def classify_font_one_image(image: np.ndarray) -> any:
    img = image.copy()
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, (1, 256, 256, 3))
    result = np.argmax(model.predict(img))
    cv2.imwrite(f"assets/{labels[result]}-{uuid.uuid4()}-image.png", image)

    # Get label names from somewhere
    return labels[result]

def classify_fonts(images: List[np.ndarray]) -> List[any]:
    output = []
    logger.info(f"Running classify_fonts on images: {len(images)}")
    for image in images:
        result = classify_font_one_image(image)
        output.append(result)

    return output