import numpy as np
import pickle
import math
import glob
import time

from tqdm import tqdm

from ase import Atom,Atoms
from ase.io import write

def compute_length(axis_val):
	non_zeros = axis_val[axis_val > 0]
	(a,) = np.where(axis_val == non_zeros.min())

	# distance from center in grid space
	N = np.abs(16 - a[0])
	
	# length of the unit vector
	r_fake = np.sqrt(-2*0.26**2*np.log(non_zeros.min())) #r_fake = N*(r/32)
	r = r_fake * 32.0 / float(N)
	return r
	
def compute_angle(ri,rj,rij):
	cos_theta = (ri**2 + rj**2 - rij**2) / (2*ri*rj)
	theta = math.acos(-cos_theta) * 180/np.pi # angle in deg.
	return theta 

def define_cell(json_name):	
	with open(json_name) as f:
		dat = json.load(f)
		img = dat['image']

	img = img.reshape(32,32,32);

	a_axis = img[:,16,16]; ra = compute_length(a_axis)
	b_axis = img[16,:,16]; rb = compute_length(b_axis)
	c_axis = img[16,16,:]; rc = compute_length(c_axis)

	ab_axis = np.array([img[i,i,16] for i in range(32)]); rab = compute_length(ab_axis)
	bc_axis = np.array([img[16,i,i] for i in range(32)]); rbc = compute_length(bc_axis)
	ca_axis = np.array([img[i,16,i] for i in range(32)]); rca = compute_length(ca_axis)

	alpha = compute_angle(rb,rc,rbc)
	beta = compute_angle(rc,ra,rca)
	gamma = compute_angle(ra,rb,rab)

	atoms = Atoms(cell=[ra,rb,rc,alpha,beta,gamma],pbc=True)
	atoms.append(Atom('Cu',[0.5]*3))
	pos = atoms.get_positions()
	atoms.set_scaled_positions(pos)
	write(pickle_name[:-7]+'.cif',atoms)

for pickle_name in tqdm(glob.glob('*.json')):
  try:
	define_cell(pickle_name)
  except:
	pass
