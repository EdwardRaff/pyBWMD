cimport cython
from cpython cimport array
import array
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

#This code is a conversion of: Kärkkäinen, J., & Sanders, P. (2003). Simple Linear Work Suffix Array Construction. In Proceedings of the 30th International Conference on Automata, Languages and Programming (pp. 943–955). Berlin, Heidelberg: Springer-Verlag. Retrieved from http://dl.acm.org/citation.cfm?id=1759210.1759301

#We are going to get a warning from cython that looks like
#warning: "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
#We can ignore it https://stackoverflow.com/questions/25789055/cython-numpy-warning-about-npy-no-deprecated-api-when-using-memoryview


cdef int leq (int a1, int a2,   int b1, int b2):
    return (a1 < b1 or (a1 == b1 and a2 <= b2))

cdef int leq3(int a1, int a2, int a3,   int b1, int b2, int b3):
    return (a1 < b1 or (a1 == b1 and leq(a2,a3, b2,b3)))

# stably sort a[0..n-1] to b[0..n-1] with keys in 0..K from r
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef void radixPass(int* a, int* b, int* r, int n, int K):
    # count occurrences
    cdef int* c = <int *>malloc((K+1) * cython.sizeof(int)) # counter array
    cdef int i
    cdef int sum
    cdef int t
    
    for i in range(K+1):
        c[i] = 0 # reset counters

    for i in range(n):
        #print(i)
        c[r[a[i]]] += 1 # count occurences
    
    sum = 0
    for i in range(K+1):
        t = c[i]
        c[i] = sum
        sum += t
    for i in range(n):
        t = c[r[a[i]]]
        b[t] = a[i];
        c[r[a[i]]] += 1
        # sort
    
    free(c)

# find the suffix array SA of s[0..n-1] in {1..K}^n
# require s[n]=s[n+1]=s[n+2]=0, n>=2
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef void suffixArray(int* s, int* SA, int n, int K):
    cdef int n0=(n+2)/3
    cdef int n1=(n+1)/3
    cdef int n2=n/3
    cdef int n02=n0+n2
    cdef int* s12 = <int *>malloc((n02 + 3) * cython.sizeof(int))
    s12[n02] = s12[n02+1] = s12[n02+2] = 0
    cdef int* SA12 = <int *>malloc((n02 + 3) * cython.sizeof(int))
    SA12[n02] = SA12[n02+1] = SA12[n02+2] = 0
    cdef int* s0 = <int *>malloc(n0 * cython.sizeof(int))
    cdef int* SA0 = <int *>malloc(n0 * cython.sizeof(int))
    
    
    #loop variables
    cdef int i
    cdef int j
    cdef int sum
    cdef int p
    cdef int t
    cdef int k
    cdef int leq_switch
    
    for i in range(n02+3):
        s12[i] = 0
        SA12[i] = 0
    for i in range(n0):
        s0[i] = 0
        SA0[i] = 0
    
    #generate positions of mod 1 and mod  2 suffixes
    #the "+(n0-n1)" adds a dummy mod 1 suffix if n%3 == 1
    j = 0
    for i in range( n+(n0-n1) ):
        if i % 3 != 0:
            s12[j] = i
            j += 1
    
    # lsb radix sort the mod 1 and mod 2 triples
    radixPass(s12 , SA12, s+2, n02, K)
    radixPass(SA12, s12 , s+1, n02, K)
    radixPass(s12 , SA12, s  , n02, K)
    
    # find lexicographic names of triples
    cdef int name = 0
    cdef int c0 = -1
    cdef int c1 = -1
    cdef int c2 = -1
    for i in range(n02):
        if s[SA12[i]] != c0 or s[SA12[i]+1] != c1 or s[SA12[i]+2] != c2 :
            name += 1
            c0 = s[SA12[i]]
            c1 = s[SA12[i]+1]
            c2 = s[SA12[i]+2]

        if SA12[i] % 3 == 1 : # left half
            s12[SA12[i]/3]      = name
        else: # right half
            s12[SA12[i]/3 + n0] = name
    
    #print("T", name, n02)
    #recurse if names are not yet unique
    if name < n02 :
        suffixArray(s12, SA12, n02, name)
        # store unique names in s12 using the suffix array
        for i in range(n02):
            s12[SA12[i]] = i + 1
    else: # generate the suffix array of s12 directly
        for i in range(n02):
            SA12[s12[i] - 1] = i
    
    # stably sort the mod 0 suffixes from SA12 by their first character
    j = 0
    for i in range(n02):
        if SA12[i] < n0:
            s0[j] = 3*SA12[i]
            j += 1
    
    radixPass(s0, SA0, s, n0, K)
    
    # merge sorted SA0 suffixes and sorted SA12 suffixes
    #for (int p=0,  t=n0-n1,  k=0;  k < n;  k++)
    p = 0
    t=n0-n1
    k=0
    while k < n:
    #define GetI() (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2)
    # GetI() =  (SA12[t] * 3 + 1 if (SA12[t] < n0) else (SA12[t] - n0) * 3 + 2)
        i =(SA12[t] * 3 + 1 if (SA12[t] < n0) else (SA12[t] - n0) * 3 + 2); # pos of current offset 12 suffix
        j = SA0[p]; # pos of current offset 0  suffix
        #if (SA12[t] < n0 ?
        #    leq(s[i],       s12[SA12[t] + n0], s[j],       s12[j/3]) :
        #    leq3(s[i],s[i+1],s12[SA12[t]-n0+1], s[j],s[j+1],s12[j/3+n0]))
        
        if SA12[t] < n0:
            leq_switch = leq(s[i],       s12[SA12[t] + n0], s[j],       s12[j/3])
        else:
            leq_switch = leq3(s[i],s[i+1],s12[SA12[t]-n0+1], s[j],s[j+1],s12[j/3+n0])
        if leq_switch:
        # suffix from SA12 is smaller
            SA[k] = i
            t += 1
            if t == n02: # done --- only SA0 suffixes left
                #for (k++;  p < n0;  p++, k++) boy, what an ugly loop..
                k += 1
                while p < n0:
                    SA[k] = SA0[p]
                    
                    k += 1
                    p += 1
        else:
            SA[k] = j
            p += 1
            if p == n0: # done --- only SA12 suffixes left
                #for (k++;  t < n02;  t++, k++) # boy, what an ugly loop...
                k += 1
                while t < n02:
                    SA[k] = (SA12[t] * 3 + 1 if (SA12[t] < n0) else (SA12[t] - n0) * 3 + 2)
                    
                    t += 1
                    k += 1
        k += 1 #mimicing the for loop that started this mess
    
    free(s12)
    free(SA12)
    free(SA0)
    free(s0)
    
    return
    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.initializedcheck(False)
@cython.cdivision(True)
def bytes_to_raw_vec(const unsigned char[:] input_bytes, unsigned int alphabet_size):
    
    cdef int L =  len(input_bytes)
    cdef int* s = <int *>malloc((L+3) * cython.sizeof(int))
    cdef int* SA = <int *>malloc(L * cython.sizeof(int))
    cdef unsigned char* raw_input = &input_bytes[0]
    
    cdef int pos = 0
    cdef unsigned char c
    
    cdef int prev_val
    cdef int cur_val

    for pos in range(L):
        s[pos] = raw_input[pos]
    s[L] = s[L+1] = s[L+2] = 0
    
    suffixArray(s, SA, L, alphabet_size)
    
    cdef np.ndarray[float, ndim=1, mode="c"] vec = np.full(shape=(alphabet_size*alphabet_size), fill_value=0.0, dtype=np.float32)
    
    
    prev_val = 0
    pos = 0
    while pos < L:
        #starts as an index
        cur_val = SA[pos]-1
        if cur_val < 0:
            cur_val = L-1
        #index in and get true value
        cur_val = s[cur_val]
        if pos > 1:
            vec[prev_val*alphabet_size + cur_val] += 1
        prev_val = cur_val
        
        pos += 1
    free(SA)
    free(s)
    if L > 1:
        vec /= (L-1)
    
    return vec
