import numpy as np
import matplotlib.pyplot as plt
import skimage
import math

from cp_hw2 import writeHDR


## Weight functions
z_min = 0.05
z_max = 0.95
def w_uniform(z, shutterSpeed=None) :
    result = ((z >= z_min) & (z <= z_max)).astype(np.float32)
    return result

def w_tent(z, shutterSpeed=None) :
    mask = ((z >= z_min) & (z <= z_max)).astype(np.float32)
    result = np.multiply(mask, np.minimum(z, 1 - z), dtype=np.float32)
    return result

def w_gaussian(z, shutterSpeed=None) :
    mask = ((z >= z_min) & (z <= z_max)).astype(np.float32)
    result = np.exp(-4 / 0.25 * np.power(z - 0.5, 2, dtype=np.float32), dtype=np.float32)
    result = np.multiply(mask, result, dtype=np.float32)
    return result

def w_photon(z, shutterSpeed=None) :
    if shutterSpeed is None:
        return np.ones(z.shape, dtype=np.float32)
    else :
        mask = ((z >= z_min) & (z <= z_max)).astype(np.float32)
        result = mask * shutterSpeed
        return result

def w_optimal(z, shutterSpeed=None) :
    gain = 0.09081608059983558
    additiveVar = 0.2669777268833028
    mask = ((z >= z_min) & (z <= z_max)).astype(np.float32)
    result = z * gain - additiveVar
    result = np.divide(shutterSpeed * shutterSpeed, result, dtype=np.float32)
    result = np.multiply(mask, result, dtype=np.float32)
    return result

## Linearize
# Uniform sampling, 6 samples on width, 4 samples on height, 3 channels
def linearize(images, weight) :
    print('Linearize...', end=' ', flush=True)
    imageShape = images[0].shape
    sampleInterval = 250
    numSamplesHorizontal = math.ceil(imageShape[1] / sampleInterval)
    numSamplesVertical = math.ceil(imageShape[0] / sampleInterval)
    numImages = len(images)
    numSamples = numImages * numSamplesHorizontal * numSamplesVertical * 3
    n = 256
    A = np.zeros((numSamples + n - 2, n + numSamplesHorizontal * numSamplesVertical))
    b = np.zeros((A.shape[0], 1))

    ## Populate matrix
    # Choose which weight function to use
    l = 1000
    x = 0
    for k in range(1, numImages + 1) :
        image = images[k - 1]
        shutterSpeed = pow(2, k - 1) / 2048
        logShutterSpeed = np.log(shutterSpeed)
        for j in range(numSamplesVertical) :
            for i in range(numSamplesHorizontal) :
                idx = j * numSamplesHorizontal + i
                for ch in range(3) :
                    sample = image[j * sampleInterval, i * sampleInterval, ch]
                    w = weight(np.array([sample / 255]), shutterSpeed)[0]
                    A[x, sample] = w
                    A[x, n + idx] = -w
                    b[x, 0] = w * logShutterSpeed
                    x = x + 1

    # Regularization term
    for i in range(n - 2) :
        w = weight(np.array([(i + 1) / 255]))[0]
        A[x, i] = l * w
        A[x, i + 1] = -2 * l * w
        A[x, i + 2] = l * w
        x = x + 1

    # Least-square optimization
    g = np.linalg.lstsq(A, b, rcond=None)[0]
    g = np.reshape(g[:256, :], (-1,))
    print('Done')
    return g


## Merge exposure stack
def mergeLinear(images, type, weight, g=None) :
    print('Begin mergeLinear...')

    # Normalize LDR image
    print('Normalize LDR image...', end=' ', flush=True)
    images_ldr = np.array(images, dtype=np.float32)
    if type == 'jpg' :
        images_ldr = images_ldr / 255
    elif type == 'tiff' :
        images_ldr = images_ldr / (pow(2, 16) - 1)
    print('Done')

    # Weight
    print('Compute weight...', end=' ', flush=True)
    w = np.array(images_ldr, dtype=np.float32)
    for i in range(w.shape[0]) :
        w[i] = weight(images_ldr[i], pow(2, i) / 2048)
    print('Done')

    # Linear image
    print('Covert to linear image...', end=' ', flush=True)
    images_lin = None
    if type == 'jpg' :
        images_lin = g[images].astype(np.float32)
        images_lin = np.exp(images_lin, dtype=np.float32)
    elif type == 'tiff' :
        images_lin = images_ldr
    print('Done')

    # Final computation
    print('Merge...', end=' ', flush=True)
    denom = np.sum(w, axis=0, dtype=np.float32)
    nomin = np.multiply(w, images_lin, dtype=np.float32)
    for i in range(images.shape[0]) :
        nomin[i] = nomin[i] / (pow(2, i) / 2048)
    nomin = np.sum(nomin, axis=0)
    image_hdr = np.divide(nomin, denom, dtype=np.float32)
    print('Done')
    print('mergeLinear finished!')
    return image_hdr

def mergeLogarithmic(images, type, weight, g=None) :
    print('Begin mergeLogarithmic...')

    # Normalize LDR image
    print('Normalize LDR image...', end=' ', flush=True)
    images_ldr = np.array(images, dtype=np.float32)
    if type == 'jpg' :
        images_ldr = images_ldr / 255
    elif type == 'tiff' :
        images_ldr = images_ldr / (pow(2, 16) - 1)
    print('Done')

    # Weight
    print('Compute weight...', end=' ', flush=True)
    w = np.array(images_ldr, dtype=np.float32)
    for i in range(w.shape[0]) :
        w[i] = weight(images_ldr[i], pow(2, i) / 2048)
    print('Done')

    # Linear image
    print('Covert to linear image...', end=' ', flush=True)
    images_lin = None
    if type == 'jpg' :
        images_lin = g[images].astype(np.float32)
        images_lin = np.exp(images_lin, dtype=np.float32)
    elif type == 'tiff' :
        images_lin = images_ldr
    print('Done')

    # Final computation
    print('Merge...', end=' ', flush=True)
    epsilon = 0.01
    denom = np.sum(w, axis=0)
    logShutterSpeed = np.zeros(images.shape, dtype=np.float32)
    for i in range(images.shape[0]) :
        logShutterSpeed[i, :, :, :] = np.log(pow(2, i) / 2048)
    nomin = np.log(images_lin + epsilon, dtype=np.float32) - logShutterSpeed
    nomin = np.multiply(w, nomin, dtype=np.float32)
    nomin = np.sum(nomin, axis=0)
    image_hdr = np.exp(np.divide(nomin, denom), dtype=np.float32)
    print('Done')
    print('mergeLogarithmic finished!')
    return image_hdr


## Select one or more from each one below
# Choose image type here
types = ['jpg', 'tiff']
typeSelect = [0, 1]
# Choose weight function here
weightFuncs = [w_uniform, w_tent, w_gaussian, w_photon, w_optimal]
weightSelect = [0, 1, 2, 3]
# Choose merge method here
mergeAlgs = [mergeLinear, mergeLogarithmic]
mergeAlgSelect = [0, 1]

for x in typeSelect :
    type = types[x]
    images = None
    for i in range(1, 17) :
        filename = '../data/door_stack/exposure' + str(i) + '.' + type
        image = skimage.io.imread(filename)
        if i == 1 :
            images = np.zeros((16, image.shape[0], image.shape[1], image.shape[2]), dtype=np.int32)
        images[i - 1] = image

    for y in weightSelect :
        weight = weightFuncs[y]
        g = None
        if type == 'jpg' :
            g = linearize(images, weight)
            # plt.plot(g, 'o', markersize=1)
            # plt.show()
        
        for z in mergeAlgSelect:
            mergeAlg = mergeAlgs[z]
            image_hdr = mergeAlg(images, type, weight, g)

            name1 = ''
            if type == 'jpg' :
                name1 = '_rendered'
            elif type == 'tiff' :
                name1 = '_raw'

            name2 = ''
            if mergeAlg == mergeLinear :
                name2 = '_lin'
            elif mergeAlg == mergeLogarithmic :
                name2 = '_log'

            name3 = ''
            if weight == w_uniform :
                name3 = '_uniform'
            elif weight == w_tent :
                name3 = '_tent'
            elif weight == w_gaussian :
                name3 = '_gaussian'
            elif weight == w_photon :
                name3 = '_photon'

            filename = 'doorstack' + name1 + name2 + name3 + '.HDR'
            writeHDR(filename, image_hdr)
            print('Written to ' + filename)


