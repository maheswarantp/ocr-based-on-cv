import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List

def plot(img: np.ndarray):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)

def extract_horizontal_text_sections(image: np.ndarray, plot_graph: bool = False) -> List[np.ndarray]:
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Vertical projection to sum pixels along vertical axes -> basically checks to see
    # if any row is full of white pixels

    vertical_projection = np.sum(thresh, axis=1)

    # Smooth with a normalized uniform box filter, simplest to work with and easiest to implement
    smoothed_projection = np.convolve(vertical_projection, np.ones((5,))/5, mode='same')


    threshold = 0.1 * np.max(smoothed_projection)
    start = 0
    sections = []
    in_text_section = False

    for i, value in enumerate(smoothed_projection):
        if value > threshold and not in_text_section:
            start = i
            in_text_section = True
        elif value <= threshold and in_text_section:
            sections.append(image[start: i, :, :])
            in_text_section = False

    # In case the last section does not end
    if in_text_section:
        sections.append(image[start : len(smoothed_projection), :, :])
    
    valid_sections = [i for i in sections if i.shape[0] >= 10]

    if plot_graph:
        for index, section in enumerate(valid_sections):
            plot(section)
            plt.savefig(f"assets/horizontal_section_split_{index}.png")
    
    return valid_sections

# Calculate distance between 2 centroids
def calculate_distance(centroid1, centroid2):
    return ((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)**0.5


def merge_boxes(image: np.ndarray):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape

    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= 20]    # 20 so that the area under consideration isnt too small

    centroids = [(0.5 * (bbox[0] + bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[1] + bbox[3])) for bbox in bounding_boxes]

    sorted_combined = sorted(zip(centroids, bounding_boxes), key = lambda x: x[0][0])
    bounding_boxes = [j for i, j in sorted_combined]
    centroids = [i for i, j in sorted_combined]


    bbox_substrings = []
    number_of_boxes = 12                            # there are 12 contours needed assuming perfect detection, which has been observed empirically
    for i in range(len(bounding_boxes) // number_of_boxes):
        x1 = bounding_boxes[i * number_of_boxes : i * number_of_boxes + number_of_boxes][0][0]
        x2 = bounding_boxes[i * number_of_boxes : i * number_of_boxes + number_of_boxes][-1][0] + bounding_boxes[i * number_of_boxes : i * number_of_boxes + number_of_boxes][-1][2]
        # bbox_substrings.append((x1 - 20, 0, abs(x2 - x1) + 20, height))
        bbox_substrings.append((x1, 0, abs(x2 - x1), height))

    return bbox_substrings

def extract_sentence(image: np.ndarray, show_strips: bool = False):
    img = image.copy()
    image_strips = []
    bbox_substrings = merge_boxes(img)
    for x, y, w, h in bbox_substrings:
        image_strips.append(image[y:y+h, x:x+w])

    if show_strips:
        for index, image_strip in enumerate(image_strips):
            plot(image_strip)
            plt.savefig(f"assets/extracted_sentence_{index}.png")

    return image_strips