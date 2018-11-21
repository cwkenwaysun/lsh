from im2hist import get_bgr_hist, connect, im_request, hist_64
from tsv_parser import im_list
import cv2
import threading

class Image:
    def __init__(self, ls):
        self.ls = ls
        self.db = connect()
        self.col = self.db.images
        self.mutex = threading.Lock()

    def im_request_worker(self, s, e):
        for i in range(s, e):
            l = self.ls[i]
            print(l, i)
            hists = []
            try:
                hists = hist_64(im_request(l))
            except:
                self.mutex.acquire()
                f = open("failed_links.txt".format(i), "a")
                f.write(str(i) + ' ')
                f.write(l + '\n')
                f.close()
                self.mutex.release()

            print(hists)
            if hists and len(hists) == 64:
                self.col.insert_one({'id': i, 'im_url': l, 'hists': hists})


if __name__ == '__main__':

    # read tsv files
    ls = []
    for i in range(10):
        ls += im_list('data/open-images-dataset-train{:}.tsv'.format(i))
    images = Image(ls)

    thread1 = threading.Thread(target=images.im_request_worker, args=(17306, 20000))
    #thread2 = threading.Thread(target=images.im_request_worker, args=(25757, 30000))
    #thread3 = threading.Thread(target=images.im_request_worker, args=(34787, 40000))
    #thread4 = threading.Thread(target=images.im_request_worker, args=(43931, 50000))
    #thread5 = threading.Thread(target=images.im_request_worker, args=(54073, 60000))

    thread1.start()
    #thread2.start()
    #thread3.start()
    #thread4.start()
    #thread5.start()
    thread1.join()
    #thread2.join()
    #thread3.join()
    #thread4.join()
    #thread5.join()
