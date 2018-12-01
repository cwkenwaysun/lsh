from lshash import LSHash
from im2hist import connect
from operator import itemgetter


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
    for r in result:
        approx_urls.append(r[0][1])

    k = len(actual)
    recall = len(intersect(actual_urls, approx_urls)) / k

    # Get Error Ratio:
    error_ratio = 0
    for i in range(0,k):
        actual_distance = actual[i][1]
        approx_distance = result[i][1]
        if (approx_distance != 0):
            error_ratio += (actual_distance / approx_distance)

    error_ratio = error_ratio / (k-1)
    print("Recall: ", recall, "  Error Ratio", error_ratio)

def find_KNN(query, k):
    db = connect()

    mongo = {"mongo": {"db": db}}

    urlandDist = []
    i = 0
    col = db.images2
    for im in col.find()[:50000]:
        if i < k:
            urlandDist.append([im['im_url'], LSHash.euclidean_dist_square(query, im['hists'])])
            i = i+1

        else:
            urlandDist.sort(key=itemgetter(1))
            if (LSHash.euclidean_dist_square(query, im['hists']) < urlandDist[k-1][1]):
                urlandDist[k-1] = [im['im_url'], LSHash.euclidean_dist_square(query, im['hists'])]
    print("Actual K nearest neighbors:")
    for r in urlandDist:
        print(r[0], r[1])

    return urlandDist




