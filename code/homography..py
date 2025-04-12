import numpy as np
import matplotlib.pyplot as plt
import os
from utils import imread
from detectBlobs import detectBlobs
from cv2 import warpPerspective as wrap
from computeSift import compute_sift
from computeMatches import computeMatches
import random


def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1] * p1[0], -p2[1] * p1[1], -p2[1] * p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1], -p2[0] * p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H / H[2, 2]
    return H


def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx]
    return np.array(point)


def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp / temp[2])[0:2]

    errors = np.linalg.norm(all_p2 - estimate_p2, axis=1) ** 2
    return errors


def ransac(matches):
    num_best_inliers = 0
    best_inliers = []
    best_H = np.array([])
    for i in range(2000):
        points = random_point(matches)
        H = homography(points)

        #  avoid dividing by zero
        if np.linalg.matrix_rank(H) < 3:
            continue

        errors = get_error(matches, H)
        idx = np.where(errors < 0.8)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()

    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H


def mergeImages(left, right, H):

    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)

    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    warped_l = wrap(src=left, M=H, dsize=size)
    height_r, width_r, channel_r = right.shape

    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)

    warped_r = wrap(src=right, M=translation_mat, dsize=size)
    black = np.zeros(3)
    for i in range(warped_r.shape[0]):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]

            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass

    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image

class Params:
    def __init__(self, levels=10, initial_sigma=2, k=2 ** 0.35, threshold=0.001):
        self.levels = levels
        self.initial_sigma = initial_sigma
        self.k = k
        self.threshold = threshold

    def set_filter_method(self, filter):
        self.filter = filter


# Image directory
dataDir = os.path.join('..', 'data', 'stitching')

# Read input images
testExamples = ['car', 'book', 'eg', 'kitchen', 'park', 'table']

output_dir = "../output/extraImageStitching/"

paramsMap = {
    'car': Params(threshold=0.002),  # Validated
    'book': Params(threshold=0.0005),  # Validated
    'eg': Params(),  # Validated
    'kitchen': Params(threshold=0.00001),  # Best possible with maxInlierError = 40,
    # randomSeedSize & goodFitThresh = 5
    'park': Params(threshold=0.0005),  # Validated
    'table': Params(threshold=0.00001),  # Validated
}

for example in testExamples:
    print(f"Stitching {example}...")
    output_path = output_dir

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    imageName1 = '{}_1.jpg'.format(example)
    imageName2 = '{}_2.jpg'.format(example)

    im1 = imread(os.path.join(dataDir, imageName1))
    im2 = imread(os.path.join(dataDir, imageName2))

    params = paramsMap[example]
    params.set_filter_method('DOG')

    # Detect keypoints
    print(f"Finding blobs...")

    blobs1 = detectBlobs(im1, params)
    blobs2 = detectBlobs(im2, params)

    # Compute SIFT features
    sift1 = compute_sift(im1, blobs1[:, :4])
    sift2 = compute_sift(im2, blobs2[:, :4])

    print(f"Finding matches and running ransac...")
    # Find the matching between features
    matches = computeMatches(sift1, sift2)

    # Ransac to find correct matches and compute transformation
    pointMatches = []
    for ind in range(matches.shape[0]):
        if matches[ind] != -1:
            point1 = blobs1[ind][:2]
            point2 = blobs2[matches[ind]][:2]
            pointMatches.append([point1[0], point1[1], point2[0], point2[1]])

    pointMatches = np.array(pointMatches)
    _, transf = ransac(pointMatches)

    tranf_print = np.round(np.linalg.inv(transf), 4)
    print(f"Estimated transformation:\n{tranf_print[:1, :].tolist()[0]}\n{tranf_print[-1:, :].tolist()[0]}")

    # Merge two images and display the output
    stitchIm = mergeImages(im1, im2, transf)

    plt.figure()
    plt.imshow(stitchIm)
    plt.title('Stitched image: {}'.format(example))
    plt.axis('off')
    plt.savefig(output_path + example + '_stitched', bbox_inches='tight', edgecolor='auto')
    plt.show()

    print(f"Completed!\n")
