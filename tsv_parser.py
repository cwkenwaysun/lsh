def im_list(filename):
    with open(filename) as f:
        # skip first line
        f.readline()
        return [l.split()[0] for l in f.readlines()]



if __name__ == '__main__':
    l = im_list('open-images-dataset-train0.tsv')
    print(l)