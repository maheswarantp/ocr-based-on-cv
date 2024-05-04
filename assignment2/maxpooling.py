import numpy as np

def maxpooling(image, window_size):
    """
        Does MaxPooling on a given image array

        Args:
            image: numpy ndarray of shape mxm
            window_size: window_size of type int
        
        Returns:
            result: numpy ndarray of shape mxm
    """
    # @TODO: Implement padding

    assert image.shape[0] == image.shape[1], "Image must be of shape mxm"
    m = image.shape[0]

    result = np.zeros((m, m), dtype = image.dtype)

    for i in range(m - window_size + 1):
        for j in range(m - window_size + 1):
            window = image[i: i + window_size, j: j + window_size]
            result[i, j] = np.max(window)


    return result

image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
result = maxpooling(image, 2)
print(result)
print(result.shape)