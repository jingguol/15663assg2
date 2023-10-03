import numpy as np
import matplotlib.pyplot as plt
import skimage
import math

from cp_hw2 import read_colorchecker_gm
from cp_hw2 import lRGB2XYZ
from cp_hw2 import XYZ2lRGB
from cp_hw2 import readHDR
from cp_hw2 import writeHDR
from cp_hw2 import xyY_to_XYZ


r, g, b = read_colorchecker_gm()
r = r.flatten()
g = g.flatten()
b = b.flatten()
checkerTruth = np.stack((r, g, b), axis=-1)
# print(checkerTruth)

## Manually crop squares within colorchecker patches for only once
# image = readHDR('doorstack_rendered_log_tent.HDR')
image = readHDR('campus_log_optimal.HDR')
# plt.imshow(np.clip(image * 50, 0.0, 1.0))
# plt.show()
# coords = plt.ginput(24 * 2, timeout=-1)
# coords = np.array(coords, dtype=np.int32)
# coords = np.reshape(coords, (24, 2, 2))
# np.save('colorchecker_coords.npy', coords)

# ## For future runs, read the square coordinates from saved file
# checkerCoords = np.load('colorchecker_coords.npy')
# checkerAvg = np.zeros((24, 3))
# for i in range(24) :
#     x0, y0 = checkerCoords[i, 0, 0], checkerCoords[i, 0, 1]
#     x1, y1 = checkerCoords[i, 1, 0], checkerCoords[i, 1, 1]
#     checkerAvg[i] = np.average(image[y0:y1+1, x0:x1+1, :], axis=(0, 1))

# ## Populate matrix and least-square optimize
# A = np.zeros((24 * 3, 12), dtype=np.float32)
# b = np.zeros((A.shape[0], 1), dtype=np.float32)
# x = 0
# for i in range(24) :
#     A[x, 0:3] = checkerAvg[i]
#     A[x, 3] = 1
#     b[x] = checkerTruth[i, 0]
#     x = x + 1
#     A[x, 4:7] = checkerAvg[i]
#     A[x, 7] = 1
#     b[x] = checkerTruth[i, 1]
#     x = x + 1
#     A[x, 8:11] = checkerAvg[i]
#     A[x, 11] = 1
#     b[x] = checkerTruth[i, 2]
#     x = x + 1
# a = np.linalg.lstsq(A, b, rcond=None)[0]
# a = a.flatten()
# A = np.array([a[0:4], a[4:8], a[8:12], [0, 0, 0, 1]])

# ## Apply affine matrix to image and clip
# image_homo = np.concatenate((image, np.ones((image.shape[0], image.shape[1], 1))), axis=2, dtype=np.float32)
# image_homo = np.expand_dims(image_homo, axis=3)
# image_transformed = np.matmul(A, image_homo, dtype=np.float32)
# image_transformed = np.squeeze(image_transformed, axis=3)
# image_transformed = image_transformed[:, :, 0:3]
# image_transformed = np.clip(image_transformed, 0.0, None, dtype=np.float32)
# # print(image_transformed.shape)
# # plt.imshow(np.clip(image_transformed, 0.0, 1.0))
# # plt.show()

# ## Apply white balancing such that the white patch is 'white'
# ## I choose not to do it for best-looking images
# x0, y0 = checkerCoords[18, 0, 0], checkerCoords[18, 0, 1]
# x1, y1 = checkerCoords[18, 1, 0], checkerCoords[18, 1, 1]
# whitePatchAvg = np.average(image[y0:y1+1, x0:x1+1, :], axis=(0, 1))
# whiteBalanceScale = whitePatchAvg[1] / whitePatchAvg
# # Uncomment this line to bring back white balancing
# # image_transformed = np.multiply(image_transformed, whiteBalanceScale, dtype=np.float32)

# # writeHDR('doorstack_colorcorrected.HDR', image_transformed)
# # fig, ax = plt.subplots(1, 2)
# # ax[0].imshow(np.clip(image_transformed, 0.0, 1.0))
# # ax[1].imshow(np.clip(image_transformed * 0.02, 0.0, 1.0))
# # plt.show()


## Tonemapping
epsilon = 1e-10
def tonemap_RGB(image, K, B) :
    I_m_hdr = np.log(image + epsilon, dtype=np.float32)
    I_m_hdr = np.sum(I_m_hdr) / (image.shape[0] * image.shape[1] * image.shape[2])
    I_m_hdr = np.exp(I_m_hdr)
    I_tilde = K / I_m_hdr * image
    I_white = B * np.max(I_tilde)
    nomin = 1 + I_tilde / (I_white * I_white)
    nomin = np.multiply(I_tilde, nomin, dtype=np.float32)
    denom = 1 + I_tilde
    I_tm = np.divide(nomin, denom, dtype=np.float32)
    return I_tm

def tonemap_xyY(image, K, B) :
    image_XYZ = lRGB2XYZ(image)
    image_xyY = np.array(image_XYZ, dtype=np.float32)
    image_xyY[:, :, 0] = np.divide(image_xyY[:, :, 0], np.sum(image_XYZ, axis=2), dtype=np.float32)
    image_xyY[:, :, 1] = np.divide(image_xyY[:, :, 1], np.sum(image_XYZ, axis=2), dtype=np.float32)
    image_xyY[:, :, 2] = image_XYZ[:, :, 1]
    I_m_hdr = np.log(image_xyY[:, :, 2] + epsilon, dtype=np.float32)
    I_m_hdr = np.sum(I_m_hdr) / (image.shape[0] * image.shape[1])
    I_m_hdr = np.exp(I_m_hdr)
    I_tilde = K / I_m_hdr * image_xyY[:, :, 2]
    I_white = B * np.max(I_tilde)
    nomin = 1 + I_tilde / (I_white * I_white)
    nomin = np.multiply(I_tilde, nomin, dtype=np.float32)
    denom = 1 + I_tilde
    image_xyY[:, :, 2] = np.divide(nomin, denom, dtype=np.float32)
    image_XYZ[:, :, 0], image_XYZ[:, :, 1], image_XYZ[:, :, 2] = xyY_to_XYZ(image_xyY[:, :, 0], image_xyY[:, :, 1], image_xyY[:, :, 2])
    I_tm = XYZ2lRGB(image_XYZ)
    return I_tm


key = 0.2
burn = 25
image_tonemapped = tonemap_xyY(image, key, burn)

# Gamma correction
mask = image_tonemapped <= 0.0031308
image_tonemapped[mask] = image_tonemapped[mask] * 12.92
image_tonemapped[np.logical_not(mask)] = (1 + 0.055) * np.power(image_tonemapped[np.logical_not(mask)], 1/2.4) - 0.055

plt.imsave('campus_optimalxyY.png', np.clip(image_tonemapped, 0.0, 1.0))
plt.imshow(np.clip(image_tonemapped, 0.0, 1.0))
plt.show()