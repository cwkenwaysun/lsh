from multiprobe_lsh import MultiprobeLSH
from config import connect
import time
import os.path
import pickle


if not os.path.isfile('tables.pickle'):
    db = connect()
    col = db.images2
    mp = MultiprobeLSH(dim=64, l=15, m=16, w=20, t=10000)

    for im in col.find():
        mp.insert(im['hists'], im['im_url'])

    file = open('tables.pickle', 'wb')
    pickle.dump(mp, file)
    file.close()

else:
    with open('tables.pickle', 'rb') as file:
        mp = pickle.load(file)


start = time.time()
res = mp.query([11.164474487304688, 0.05191167195638021, 0.0, 0.0, 2.5472323099772134, 1.4715194702148438,
                0.00022252400716145834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.119781494140625,
                0.0010808308919270833, 0.0, 0.0, 1.7841021219889324, 16.666793823242188, 0.32784144083658856, 0.0,
                0.0, 1.2158711751302083, 1.6774177551269531, 0.0007947285970052084, 0.0, 0.0, 0.0,
                0.000031789143880208336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.101470947265625, 0.00006357828776041667, 0.0,
                0.0, 36.63552602132162, 20.49719492594401, 0.010585784912109375, 0.0, 0.0, 0.9306589762369791,
                1.5670458475748699, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.021870930989583332, 0.0,
                0.0, 0.0, 0.055631001790364586, 3.1508763631184897], 10)
end = time.time()
print("time:", end - start)
for r in res:
    print(r)


# https://farm1.staticflickr.com/3196/2517231702_a53183f28f_o.jpg
start = time.time()
res = mp.query([8.441666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 8.952083333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 33.245416666666664, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 49.36083333333333], 10)
end = time.time()
print("time:", end - start)
for r in res:
    print(r)


# https://farm7.staticflickr.com/477/19709173536_4378e977eb_o.jpg
start = time.time()
res = mp.query([21.991979897879595, 1.2493254702436016, 0.32390182021004515, 1.0012799060419728, 0.03204016342892384,
     2.4937336752462316, 0.7446468411601458, 0.3063300366400029, 0.0, 0.0013603961473581108, 0.7374905905933141,
     39.71947214362156, 0.0, 0.0, 2.5507427762964577E-4, 8.455216325660698, 0.2843227947978451, 0.06620594583809472,
     0.0013603961473581108, 5.526609348642324E-4, 0.03671652518546734, 3.4722552874063592, 0.4420720647185794,
     0.06522816110718108, 0.0, 0.07418410241062197, 0.6131843926284667, 0.1532712992690138, 0.0, 0.0,
     0.0014170793201646987, 3.1230586013313744, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04463799858518801, 0.008870916544231014,
     4.251237960494096E-5, 0.0, 0.11870873465019681, 2.8496756588852006, 0.3437834430719559, 0.0, 0.0,
     0.0557053880756743, 1.149279670239974, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.043461822749451304,
     0.002763304674321162, 0.0, 0.0, 0.0613453637699298, 9.930863534127806], 10)
end = time.time()
print("time:", end - start)
for r in res:
    print(r)

# https://farm8.staticflickr.com/8713/16147802583_f7872034ec_o.jpg
start = time.time()
res = mp.query([42.36157417419582, 2.8210929490260157, 0.004029900358836345, 0.0, 3.63838677832244, 16.317717344290656,
     0.20740220107650903, 1.8772827758554403E-4, 0.0, 0.3723027201076509, 0.510821158528771, 0.01914828431372549, 0.0,
     0.0, 0.006996007144687941, 0.014943170895809304, 1.2883416234140714, 0.08810713828014866, 0.008685561642957836,
     7.50913110342176E-5, 0.6565108171536589, 7.8752638208061, 1.703634119088812, 0.016582664520056387,
     1.2515218505702935E-5, 0.9949974168589004, 6.092971553408946, 0.17968099208637703, 0.0, 0.0, 0.11146053601179033,
     0.26339528867102396, 2.3778915160835575E-4, 5.381543957452262E-4, 0.0, 0.0, 0.13060882032551582,
     1.1635899253492246, 0.06048605103806228, 0.07870820918236576, 0.009861992182493912, 0.586951232698962,
     3.2883236014994233, 1.0968462650583108, 0.0, 3.75456555171088E-5, 0.2110941905356914, 1.8419773444508523, 0.0, 0.0,
     0.0, 0.0, 2.503043701140587E-5, 0.0018897979943611433, 4.1300221068819686E-4, 0.0, 0.0, 0.10422673971549404,
     0.22026784570037164, 0.10676732907215174, 1.2515218505702935E-5, 0.02937321783288479, 0.15169696350762527,
     5.36174488177624], 10)
end = time.time()
print("time:", end - start)
for r in res:
    print(r)





