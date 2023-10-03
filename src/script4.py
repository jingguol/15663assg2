import numpy as np
import matplotlib.pyplot as plt
import skimage
import math


numImages = 50
## Compute dark frame and save it to avoid repeated calculations
# darkFrame = None
# for i in range(numImages):
#     image = plt.imread('../data/dark/' + 'DSC_' + str(i).zfill(4) + '.tiff')
#     if i == 0 :
#         darkFrame = np.zeros(image.shape, dtype=np.float32)
#     darkFrame = darkFrame + image
# darkFrame = darkFrame / numImages
# np.save('darkFrame.npy', darkFrame)

## Then we can just read from the saved darkFrame
darkFrame = np.load('darkFrame.npy')


## Average of ramp image after dark fram subtraction
## Again, save it to avoid repeated calculations in future runs
# rampAvg = np.zeros(darkFrame.shape, dtype=np.float32)
# pixel1 = np.zeros(50, dtype=np.float32)     # Pixel 1 is (2000, 1500, 0)
# pixel2 = np.zeros(50, dtype=np.float32)     # Pixel 2 is (2000, 3000, 1)
# pixel3 = np.zeros(50, dtype=np.float32)     # Pixel 3 is (2000, 4500, 2)
# for i in range(numImages):
#     image = plt.imread('../data/ramp/' + 'DSC_' + str(i).zfill(4) + '.tiff')
#     image = image - darkFrame
#     rampAvg = rampAvg + image
#     # Record some pixel values for the histogram
#     pixel1[i] = image[2000, 1500, 0]
#     pixel2[i] = image[2000, 3000, 1]
#     pixel3[i] = image[2000, 4500, 2]
# rampAvg = rampAvg / numImages
# np.save('rampAvg.npy', rampAvg)

# Draw histogram
# fig, ax = plt.subplots(1, 3)
# ax[0].hist(pixel1, bins = list(range(int(pixel1.min()), int(pixel1.max() + 1), 1)))
# ax[1].hist(pixel2, bins = list(range(int(pixel2.min()), int(pixel2.max() + 1), 1)))
# ax[2].hist(pixel3, bins = list(range(int(pixel3.min()), int(pixel3.max() + 1), 1)))
# plt.show()

## Just read from saved rampAvg
rampAvg = np.load('rampAvg.npy')


## Variance of ramp image
# rampVar = np.zeros(darkFrame.shape, dtype=np.float32)
# for i in range(numImages):
#     image = plt.imread('../data/ramp/' + 'DSC_' + str(i).zfill(4) + '.tiff')
#     if i == 0 :
#         plt.imshow(image)
#         plt.show()
#     diff = np.subtract(image, darkFrame, dtype=np.float32)
#     diff = np.subtract(diff, rampAvg, dtype=np.float32)
#     diff = np.multiply(diff, diff, dtype=np.float32)
#     rampVar = rampVar + diff
# rampVar = rampVar / (numImages - 1)
# np.save('rampVar.npy', rampVar)

## Just read
rampVar = np.load('rampVar.npy')


rampAvg = np.round(rampAvg)
x = np.unique(rampAvg)
y = []
for i in x :
    mask = (rampAvg == i).astype(np.int32)
    count = np.sum(mask)
    sum = np.sum(mask * rampVar)
    y.append(sum / count)
y = np.array(y)
fig, ax = plt.subplots()
ax.plot(x, y, 'b')
gain, additiveVar = np.polyfit(x, y, 1)
print(gain, additiveVar)
x1 = np.linspace(np.min(x), np.max(x), 100)
y1 = gain * x1 + additiveVar
ax.plot(x1, y1, 'r')
plt.show()