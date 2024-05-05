import numpy as np

def maxpool2d(image, k, s, padding = 'valid'):
    m = image.shape[0]
    if padding == 'same':
        output_size = m
        
        # op = int((m - k + 2p) / s) + 1
        # 2p = max((op - 1) * s + k - m, 0)

        pad_total = max(((output_size - 1) * s) + k - m, 0)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_bottom = pad_total // 2
        pad_top = pad_total - pad_bottom

        padded_matrix = np.zeros((m + pad_top + pad_bottom, m + pad_left + pad_right))
        padded_matrix[pad_top : pad_top + m, pad_left : pad_left + m] = image
    else:
        output_size = ((m - k) // s) + 1
        padded_matrix = image

    result = np.zeros((output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            window = padded_matrix[i * s : (i * s) + k, j * s : (j * s) + k]
            result[i, j] = np.max(window)
    
    return result

image = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])

print(maxpool2d(image, k = 2, s = 1, padding = 'valid'))
print(maxpool2d(image, k = 2, s = 1, padding = 'valid'))
print(maxpool2d(image, k = 3, s = 1, padding = 'valid'))

print(maxpool2d(image, k = 2, s = 1, padding = 'same'))
