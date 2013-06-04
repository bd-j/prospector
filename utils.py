import glob
import numpy as np

def load_angst_sfh(name, sfhdir = 'data/sfhs/'):
    ty = '<f8'
    dt = np.dtype([('t1', ty), ('t2',ty), ('dmod',ty), ('sfr',ty), ('met', ty)])
    fn = glob.glob("{0}*{1}*sfh".format(sfhdir,name))
    data = np.loadtxt(fn[0], usecols = (0,1,2,3,6) ,dtype = dt)

    return data

