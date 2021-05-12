import cv2
import matplotlib.pyplot as plt
import numpy as np
from ImageCutter import CutImage, NoiseEstimation
from PIL import Image


def MergeBlock(kwargs):
    blockMatrix = kwargs["blk"]
    label = kwargs["label"]
    label_idx = kwargs["label_id"]
    HH = kwargs["HH"]
    img = kwargs["img"]
    blocksize = kwargs["BLOCK"]
    heatmap = np.array([NoiseEstimation(blockMatrix[i]) for i in range(len(blockMatrix))]).reshape(
        int(HH.shape[0] / blocksize),
        int(HH.shape[1] / blocksize))
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap, cmap="gray")
    heatmap = heatmap.flatten()
    THRESH = 0.15
    print("////***********************PROGRAM**HAS BEEN**STARTED*****************************////")
    for i in range(len(heatmap) - 1):
        for j in range(i + 1, len(heatmap)):
            if np.abs(heatmap[i] - heatmap[j]) < THRESH:
                label[j] = label[i]
    print("////**********************PROGRAM****FINISHED*****************************////")
    label = label.reshape(int(HH.shape[0] / blocksize), int(HH.shape[1] / blocksize))

    repeatLabelCounts = {i: np.sum(label == i) for i in label_idx}
    maxNum = 0
    maxNumIndex = 0
    for key, value in repeatLabelCounts.items():
        if value > maxNum:
            maxNum = value
            maxNumIndex = key

    for i in range(int(HH.shape[0] / blocksize)):
        for j in range(int(HH.shape[1] / blocksize)):
            if label[i][j] == maxNumIndex or repeatLabelCounts[label[i][j]] < 3:
                label[i][j] = 0
            else:
                label[i][j] = 255

    label = label.astype(np.uint8)
    label = cv2.resize(label, (img.shape[1], img.shape[0]))
    label[label < 128] = 0
    label[label >= 128] = 255
    cv2.imwrite("DETECTEDIMAGE.jpg", label)
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap="gray")
    plt.show()
    return label


if __name__ == "__main__":
    import glob

    variable_size_block = int(input("ENTER THE BLOCK SIZE:"))
    path = glob.glob("./TESTS/1920x1080/test1_fake.jpg")
    for i in range(len(path)):
        dicts = CutImage(path[i], variable_size_block, None)
        label = MergeBlock(dicts)
