import numpy as np
from typing import List
from cv.utils import extract_sentence, extract_horizontal_text_sections
import itertools
import logging

logger = logging.getLogger(__name__)

def extract_string(image: np.ndarray):
    # Pass in full image, RGB here
    logger.info("Running horizontal script extraction now")
    horizontal_sections_where_text_present = extract_horizontal_text_sections(image=image, plot_graph=True)
    extracted_substrings_overall = []
    bbox_substrings = []
    for horizontal_section in horizontal_sections_where_text_present:
        # Now get each instance of hello world string in one horizontal strip
        extracted_substrings, bbox_substring = extract_sentence(horizontal_section[0], show_strips=True)
        bbox_substring.append(horizontal_section[1:])
        extracted_substrings_overall.append(extracted_substrings)
        bbox_substrings.append((bbox_substring))

    return list(itertools.chain(*extracted_substrings_overall)), bbox_substrings

    