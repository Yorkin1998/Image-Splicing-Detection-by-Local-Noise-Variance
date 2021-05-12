import cv2
import numpy as np
import pywt


def CutImage(path, variable_size_block, maskpath):
    img = cv2.imread(path, 0)
    print(img.shape)
    h,w = img.shape
    if h < 300 and w < 300:
        BLOCK = variable_size_block
    elif h > 1000 or w > 1000:
        BLOCK = variable_size_block
    else:
        BLOCK = variable_size_block
    print("WIDTH:", w, "HEIGHT:", h)
    coeffs2 = pywt.dwt2(img, 'db8')
    LL, (LH, HL, HH) = coeffs2
    HH = HH[0:int(np.floor(HH.shape[0] / BLOCK)) * BLOCK, 0:int(np.floor(HH.shape[1] / BLOCK)) * BLOCK]
    blockMatrix = []
    for i in range(0, HH.shape[0], BLOCK):
        for j in range(0, HH.shape[1], BLOCK):
            blockWindow = HH[i:i + BLOCK, j:j + BLOCK]
            blockMatrix.append(blockWindow)
    label = [i for i in range(len(blockMatrix))]
    blockMatrix = np.array(blockMatrix)
    label = np.array(label)
    label_idx = list(label.copy())
    dicts = {"blk": blockMatrix, "label": label, "label_id": label_idx, "HH": HH, "img": img, "BLOCK": BLOCK}
    return dicts


def NoiseEstimation(matrix):
    return np.mean(np.abs(matrix)) / 0.6745
