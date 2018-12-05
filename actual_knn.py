import numpy as np
from config import connect
from operator import itemgetter
from im2hist import hist_64, im_request
import os
import pickle


def euclidean_dist(x, y):
    """ This is a hot function, hence some optimizations are made. """
    diff = np.array(x) - y
    return np.sqrt(np.dot(diff, diff))


def euclidean_dist_square(x, y):
    """ This is a hot function, hence some optimizations are made. """
    diff = np.array(x) - y
    return np.dot(diff, diff)


def euclidean_dist_centred(x, y):
    """ This is a hot function, hence some optimizations are made. """
    diff = np.mean(x) - np.mean(y)
    return np.dot(diff, diff)


def l1norm_dist(x, y):
    return sum(abs(x - y))


def cosine_dist(x, y):
    return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)


def intersect(a, b):
    return list(set(a) & set(b))


# Result = vector of the form [ [hist, url], [distance] ]
def accuracy(q, k, result):
    # Get "Recall":
    actual = find_KNN(q, k)

    # Convert to list of urls
    actual_urls = []
    for r in actual:
        actual_urls.append(r[0])


    approx_urls = []
    result_with_dist = []
    for r in result:
        approx_urls.append(r)
        result_with_dist.append([r, euclidean_dist(q, hist_64(im_request(r)))])

    k = len(actual)
    recall = len(intersect(actual_urls, approx_urls)) / k

    # Get Error Ratio:
    error_ratio = 0
    for i in range(0,k):
        actual_distance = actual[i][1]
        approx_distance = 1
        if len(result_with_dist) > i:
            approx_distance = result_with_dist[i][1]
        if (approx_distance != 0):
            error_ratio += (actual_distance / approx_distance)

    error_ratio = error_ratio / (k-1)
    return [recall, error_ratio]

def find_KNN(query, k):

    urlandDist = []
    i = 0

    if not os.path.isfile('raws.pickle'):
        db = connect()
        col = db.images2
        raws = list(col.find())
        file = open('raws.pickle', 'wb')
        pickle.dump(raws, file)
        file.close()

    else:
        with open('raws.pickle', 'rb') as file:
            raws = pickle.load(file)

    for im in raws:
        if i < k:
            urlandDist.append([im['im_url'], euclidean_dist_square(query, im['hists'])])
            i = i+1

        else:
            urlandDist.sort(key=itemgetter(1))
            if (euclidean_dist_square(query, im['hists']) < urlandDist[k-1][1]):
                urlandDist[k-1] = [im['im_url'], euclidean_dist_square(query, im['hists'])]
    print("Actual K nearest neighbors:")
    for r in urlandDist:
        print(r[0], r[1])

    return urlandDist




