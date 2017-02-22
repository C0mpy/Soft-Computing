import HogDescriptor as h
from sklearn.svm import SVC
from skimage.io import imread
import time
import numpy as np
import os
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    start = time.time()
    descriptor = h.HOGDescriptor((5, 5))

    pos_imgs = []
    neg_imgs = []
    test = []
    labels = []
    test_labels = []

    pos_dir = os.listdir("../p/")
    neg_dir = os.listdir("../n/")

    for i in range(50):
        if i < 30:
            pos_imgs.append(imread("../p/" + pos_dir[i]))
            neg_imgs.append(imread("../n/" + neg_dir[i]))
        else:
            if i % 2:
                test.append(imread("../p/" + pos_dir[i]))
                test_labels.append(1)
            else:
                test.append(imread("../n/" + neg_dir[i]))
                test_labels.append(0)

    labels = [1] * 30 + [0] * 30
    pos_hog = []
    for img in range(len(pos_imgs)):
        pos_hog.append(descriptor.describe(pos_imgs[img]))

    neg_hog = []
    for img in range(len(neg_imgs)):
        neg_hog.append(descriptor.describe(neg_imgs[img]))

    test_hog = []
    for img in range(len(test)):
        test_hog.append(descriptor.describe(test[img]))

    pos_hog = np.array(pos_hog)
    neg_hog = np.array(neg_hog)
    test_hog = np.array(test_hog)

    x = np.vstack((pos_hog, neg_hog))
    y = np.array(labels)

    clf = SVC(kernel="linear")
    clf = clf.fit(x, y)

    y_pred = clf.predict(test_hog)
    print time.time() - start
    print str(y_pred)
    print str(test_labels)
    print "Acc:", accuracy_score(test_labels, y_pred)