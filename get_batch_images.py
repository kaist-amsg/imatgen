#!/qcfs/jackjack5/3party-software/anaconda/kdft/anaconda2/bin/python

import glob
import argparse
import os
import numpy as np
import pickle
import sys
from tqdm import tqdm
import random

def makebatch(args):
	filelist = glob.glob(args.files+'/*.json')
	batchsize = args.batchsize
	savedir = args.savedir
	batch_iter = 0
	batch_images = []
	batch_channels = []
	random.seed(777) # You can also change random seed value
	random.shuffle(filelist)
	tt = np.array(filelist)
	if not os.path.isdir(savedir):
		os.mkdir(savedir)
	sum_nc = 0
	for filename in tqdm(filelist):
		if not '.npy' in filename:
			continue
		images = np.array(json.load(open(filename))['image'])
		dim = images.shape[0]
		nc = images.shape[-1]
		sum_nc += nc
		for c in range(nc):
			batch_images.append(images[:,:,:,c].reshape(dim,dim,dim,1))
			if len(batch_images)==batchsize:
				print 'saving batch #',batch_iter
				print sum_nc
				batch_savefilename1 = savedir+'/'+str(batch_iter)+'_images.npy'
				batch_savefilename2 = savedir+'/'+str(batch_iter)+'_pvals.npy'
				fin_batch = np.array(batch_images)
				(p,q,r,s,t) = np.where(fin_batch >= 0.02)
				pvals = fin_batch[p,q,r,s,t]
				fin_batch2 = np.array([p,q,r,s,t],np.int32)
				np.save(batch_savefilename1, fin_batch2)
				np.save(batch_savefilename2, np.array(pvals))
				batch_iter += 1
				batch_images = []
				batch_channels = []
	return 1

def main():
	parser = argparse.ArgumentParser(description='script for making single image batches')
	parser.add_argument('--savedir',type=str,default='batch_images/',
		help = 'save destination for batch images')
	parser.add_argument('--files',type=str,
		help = 'input image files, xx.npy')
	parser.add_argument('--batchsize',type=int,default=20,
		help = 'the size of batches')
	args = parser.parse_args()

	makebatch(args)
	
	return



if __name__=='__main__':
	main()	


