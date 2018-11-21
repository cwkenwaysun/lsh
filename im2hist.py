import cv2
import numpy as np
import pymongo
import config
from urllib import request, error

def get_bgr_hist(im):
    """
    :param im: image object from cv2.imread
    :return: 3x256 list
    """
    res = []
    for i in [0, 1, 2]:
        hist = cv2.calcHist([im], [i], None, [256], [0.0, 255.0])
        r = np.ndarray.tolist(hist)
        flat_r = [a for n in r for a in n]
        res.append(flat_r)

    return res


def connect():
    """
    Create pymongo client and return lsh db
    """
    client = pymongo.MongoClient(config.DATABASE['link'])
    db = client.lsh
    return db


def im_request(url):
    """
    Get image array by request url. Return None if the image is not usable(PNG).
    :param url: string
    :return: None or image array
    """
    try:
        resp = request.urlopen(url)
    except error.HTTPError as e:
        print(e)
        raise

    image = None
    if resp.info()["content-type"] == "image/jpeg":
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def hist_64(im):
    """
    :param im: image array
    :return: length 64 list, mean 64 dimension histogram from im
    """
    hist = cv2.calcHist(im, [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    #hist = cv2.calcHist(im, [2], None, [256], [0.0, 255.0])
    return list(map(int, hist.flatten()))

