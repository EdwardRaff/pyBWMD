import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse

import pyximport;
pyximport.install(setup_args={"include_dirs":np.get_include()})
from . import suffixArray

import os
from multiprocessing import Pool
import functools
import scipy

def isFile(s):
    try:
        return isinstance(s, str) and os.path.isfile(s)
    except:
        return False
        
def makeMatrix(vecs):
    mostly_dense = np.sum([isinstance(v, np.ndarray) for v in vecs]) > len(vecs)/2
    #TODO, i'm sure this could be made faster/lest wastefull...
    if mostly_dense:
        vecs = [v if isinstance(v, np.ndarray) else v.todense() for v in vecs]
        return np.vstack(vecs)
    else:
        vecs = [csr_matrix(v) if isinstance(v, np.ndarray) else v for v in vecs]
        return sparse.vstack(vecs)


def vectorize(b, alphabet_size=256, processes=-1):
    if isinstance(b, list): #Assume this is a list of things to hash.
        mapfunc = functools.partial(vectorize, alphabet_size=alphabet_size)
        if processes < 0:
            processes = None
        elif processes <= 1: # Assume 0 or 1 means just go single threaded
            return makeMatrix([z for z in map(mapfunc, b)])
        #Else, go multi threaded!
        pool = Pool(processes)
        to_ret = [z for z in pool.map(mapfunc, b)]
        pool.close()
        return makeMatrix(to_ret)
    #Not a list, assume we are processing a single file
    if isFile(b): #Was b a path? If it was a valid one, lets hash that file!
        in_file = open(b, "rb") # opening for [r]eading as [b]inary
        data = in_file.read()
        in_file.close()
        b = data
    elif isinstance(b, str): #Was a string?, convert to byte array
        b = str.encode(b)
    elif not isinstance(b, bytes):
        raise ValueError('Input was not a byte array, or could not be converted to one.')
    vec = suffixArray.bytes_to_raw_vec(b, alphabet_size)
    #TODO, this could be mad emore efficent
    #Doing un-needed math on zeros
    vec = np.sqrt(vec)
    vec /= np.sqrt(2.0)
    
    #If the vec is mostly zero, return sparse version
    if np.sum(vec != 0)*2 < vec.shape[0]:
        return csr_matrix(vec)
    
    return vec
