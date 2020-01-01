# Burrows Wheeler Markov Distance

pyBWMD is an implementation of the Burrows Wheeler Markov Distance (BWMD). It is an approach inspired by the Normalized Compression Distance ([NCD](https://en.wikipedia.org/wiki/Normalized_compression_distance)). The basic goal is to use compression as a method of measuring similarity, and thus, gives us a method we can use for any possible input. Though it is not always the best approach, it is very versatile and especially useful in domains where we do not necessarily know how to extract features, such as malware analysis. 

BWMD works by using the [Burrows Wheeler Transform (BWT)](https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform), and developing a Markov model based on the BWT transform of the input data. This works because the BWT tends to create repetition in the data, allowing simple compression techniques like run length encoding to become effective. 

Unique to BWMD is that it converts the input sequence of bytes into a vector in euclidean space, making it easy to apply to all of your favorite machine learning classification, clustering, and search algorithms. Check the  `examples` directory in this repo for some small snipets showing how to get started with BWMD. 


# Insallation 

To install pyBWMD, you can currently use this syntax with pip:
```
pip install git+git://github.com/EdwardRaff/pyBWMD#egg=pyBWMD
```
 
Or, you can download the repo and run
```
python setup.py install
```

## Citations

If you use BWMD, please cite the [original paper](https://arxiv.org/pdf/1912.13046.pdf)! Please note that this implementation was not the one used in the paper, but a re-implementation I've done seperatly. Since it is in python/cython, my experience is it will be 2-10x slower than the original Java code. I'm hoping I'll get to release that soon! 

```
@inproceedings{Raff2020,
author = {Raff, Edward and Nicholas, Charles and McLean, Mark},
booktitle = {The Thirty-Fourth AAAI Conference on Artificial Intelligence},
title = {{A New Burrows Wheeler Transform Markov Distance}},
url = {http://arxiv.org/abs/1912.13046},
year = {2020}
}
```
