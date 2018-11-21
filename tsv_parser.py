def im_list(filename):
    """
    Read tsv file and return url list
    :param filename: file path + file name
    :return: list
    """
    with open(filename) as f:
        # skip first line
        f.readline()
        return [l.split()[0] for l in f.readlines()]



