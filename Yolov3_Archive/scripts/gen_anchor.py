'''
Created on Feb 20, 2017

@author: jumabek
'''
from os.path import join
# import cv2
import numpy as np
import sys
import os
import random

widthInCfgFile = 352.
heightInCfgFile = 352.


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avgIOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def writeAnchorsToFile(centroids, X, anchorFile):
    f = open(anchorFile, 'w')

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= widthInCfgFile / 32.
        anchors[i][1] *= heightInCfgFile / 32.

    widths = anchors[:, 0]
    sortedIndices = np.argsort(widths)

    print('Anchors = ', anchors[sortedIndices])

    for i in sortedIndices[:-1]:
        f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n' % (anchors[sortedIndices[-1:], 0], anchors[sortedIndices[-1:], 1]))

    f.write('%f\n' % (avgIOU(X, centroids)))
    print()


def kmeans(X, centroids, eps, anchorFile):
    N = X.shape[0]
    iterations = 0
    k, dim = centroids.shape
    prevAssignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter, np.sum(np.abs(old_D - D))))

        # assign samples to centroids 
        assignments = np.argmin(D, axis=1)

        if (assignments == prevAssignments).all():
            print("Centroids = ", centroids)
            writeAnchorsToFile(centroids, X, anchorFile)
            return

        # calculate new centroids
        centroidSums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroidSums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroidSums[j] / (np.sum(assignments == j))

        prevAssignments = assignments.copy()
        old_D = D.copy()


def main(filelist, outputDir='generated_anchors', numClusterss=0, width=352, height=352):
    widthInCfgFile = float(width)
    heightInCfgFile = float(height)
    numClusterss = int(numClusterss)
    print("Anchors are generating for ", widthInCfgFile, heightInCfgFile)
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    f = open(filelist)

    lines = [line.rstrip('\n') for line in f.readlines()]

    annotationDims = []

    size = np.zeros((1, 1, 3))
    count = 0
    for line in lines:

        line = line.replace('images', 'labels')

        line = line.replace('.jpg', '.txt')
        line = line.replace('.png', '.txt')
        print(f"\r {count}/{len(lines)}   {line}", end='')
        count += 1
        f2 = open(line)
        f2lines = f2.readlines()
        if len(f2lines) == 0:
            continue
        for line in f2lines:
            line = line.rstrip('\n')
            w, h = line.split(' ')[3:]
            # print(w,h)
            # print("DEBUG: ", tuple(map(float,(w,h))))
            annotationDims.append(tuple(map(float, (w, h))))
    annotationDims = np.array(annotationDims)
    print("annotationDims", len(annotationDims))

    eps = 0.005

    if numClusterss == 0:
        for numClusters in range(1, 11):  # we make 1 through 10 clusters 
            anchorFile = join(outputDir, 'anchors%d.txt' % (numClusters))

            indices = [random.randrange(annotationDims.shape[0]) for i in range(numClusters)]
            centroids = annotationDims[indices]
            kmeans(annotationDims, centroids, eps, anchorFile)
            print('centroids.shape', centroids.shape)
    else:
        anchorFile = join(outputDir, 'anchors%d.txt' % (numClusterss))
        indices = [random.randrange(annotationDims.shape[0]) for i in range(numClusterss)]
        centroids = annotationDims[indices]
        kmeans(annotationDims, centroids, eps, anchorFile)
        print('centroids.shape', centroids.shape)


if __name__ == "__main__":
    main(sys.argv)
