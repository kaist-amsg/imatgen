import os
import sys
import pickle
import math
import glob
import numpy as np
from tqdm import tqdm

from ase.io import read,write
from ase import Atoms,Atom

from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def detect_peaks(image):
	neighborhood = generate_binary_structure(3, 2)
	local_max = (maximum_filter(image, footprint = neighborhood, mode = "wrap") == image)

	background = (image < 0.02)

	eroded_background = binary_erosion(background, structure = neighborhood, border_value = 1)
	detected_peaks = np.logical_and(local_max, np.logical_not(eroded_background))
	return detected_peaks

def reconstruction(image,ele):
  # image should have dimension of (N,N,N)
  image0 = gaussian_filter(image,sigma=0.15)
  peaks = detect_peaks(image0)

  recon_mat = Atoms(cell=15*np.identity(3),pbc=[1,1,1])
  (peak_x,peak_y,peak_z) = np.where(peaks==1.0)
  for px,py,pz in zip(peak_x,peak_y,peak_z):
	if np.sum(image[px-3:px+4,py-3:py+4,pz-3:pz+4] > 0) >= 0:
	  recon_mat.append(Atom(ele,(px/64.0,py/64.0,pz/64.0)))
  pos = recon_mat.get_positions()
  recon_mat.set_scaled_positions(pos)
	
  return recon_mat

ele = ['O','V']; done=0;			
for cucum in tqdm(glob.glob('*.json')):	
  with open(cucum) as f:
	dat = json.load(f)
	img = dat['image']

  tmp_mat = []
  for idc in range(2):
	image = img[:,:,:,idc].reshape(64,64,64)
	tmp_mat.append(reconstruction(image,ele[idc]))

  for atom in tmp_mat[-1]:
	tmp_mat[0].append(atom)	

  write(cucum[:-7]+'.cif',tmp_mat[0])
