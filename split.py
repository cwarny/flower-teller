from glob import glob
from os import makedirs, rename, listdir
from shutil import copy, rmtree
import numpy as np

root = 'n11669921'

# Create sample split for rapid prototyping
np.random.seed = 41
for d in glob(root + '/n*'):
	files = np.array(listdir(d))[:100]
	msk = np.random.rand(len(files)) < 0.8
	trn = files[msk]
	val = files[~msk]
	train_dir = '/'.join([root, 'sample/train', d.split('/')[-1]])
	makedirs(train_dir)
	for f in trn:
		copy(d + '/' + f, train_dir)
	valid_dir = '/'.join([root, 'sample/valid', d.split('/')[-1]])
	makedirs(valid_dir)
	for f in val:
		copy(d + '/' + f, valid_dir)

# Full split
np.random.seed = 42
for d in glob(root + '/n*'):
	files = np.array(listdir(d))
	msk = np.random.rand(len(files)) < 0.8
	trn = files[msk]
	val = files[~msk]
	train_dir = '/'.join([root, 'train', d.split('/')[-1]])
	makedirs(train_dir)
	for f in trn:
		rename(d + '/' + f, train_dir + '/' + f)
	valid_dir = '/'.join([root, 'valid', d.split('/')[-1]])
	makedirs(valid_dir)
	for f in val:
		rename(d + '/' + f, valid_dir + '/' + f)
	rmtree(d)

