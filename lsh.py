from lshash import LSHash
from im2hist import connect

db = connect()

mongo = {"mongo": {"db": db}}


lsh = LSHash(64, 64, 3, storage_config=mongo)

#col = db.images
#for im in col.find()[:20]:
#    lsh.index(im['hists'], im['im_url'])
#lsh.upload(db)

res = lsh.query([1727, 17, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 12, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 2)

'''
lsh = LSHash(6, 8, 5)
lsh.index([1,2,3,4,5,6,7,8])
lsh.index([2,3,4,5,6,7,8,9])
lsh.index([10,12,99,1,5,31,2,3])
lsh.index([10,11,99,1,5,31,2,3])
res = lsh.query([1,2,3,4,5,6,7,7], 3)
'''
print(res)