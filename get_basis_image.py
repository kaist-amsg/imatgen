import sys
import math
import os
import random
import json
from ase.io import read,write
from ase import Atom, Atoms
import argparse
import numpy
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import pickle

def get_atomlist_atomindex():
#	cod_atomlist = ['Ru', 'Re', 'Ra', 'Rb', 'Rn', 'Rh', 'Be', 'Ba', 'Bi', 'Bk', 'Br', 'H', 'P', 'Os', 'Ge', 'Gd', 'Ga', 'Pr', 'Pt', 'Pu', 'C', 'Pb', 'Pa', 'Pd', 'Cd', 'Po', 'Pm', 'Ho', 'Hf', 'Hg', 'He', 'Mg', 'K', 'Mn', 'O', 'S', 'W', 'Zn', 'Eu', 'Zr', 'Er', 'Ni', 'Na', 'Nb', 'Nd', 'Ne', 'Np', 'Fe', 'B', 'F', 'Sr', 'N', 'Kr', 'Si', 'Sn', 'Sm', 'V', 'Sc', 'Sb', 'Se', 'Co', 'Cm', 'Cl', 'Ca', 'Cf', 'Ce', 'Xe', 'Tm', 'Cs', 'Cr', 'Cu', 'La', 'Li', 'Tl', 'Lu', 'Th', 'Ti', 'Te', 'Tb', 'Tc', 'Ta', 'Yb', 'Dy', 'I', 'U', 'Y', 'Ac', 'Ag', 'Ir', 'Am', 'Al', 'As', 'Ar', 'Au', 'In', 'Mo'] # 96

#	mp_atomlist = ['Ru', 'Re', 'Rb', 'Rh', 'Be', 'Ba', 'Bi', 'Br', 'H', 'P', 'Os', 'Ge', 'Gd', 'Ga', 'Pr', 'Pt', 'Pu', 'Mg', 'Pb','Pa', 'Pd', 'Cd', 'Pm', 'Ho', 'Hf', 'Hg', 'He', 'C', 'K', 'Mn', 'O', 'S', 'W', 'Zn', 'Eu', 'Zr', 'Er', 'Ni', 'Na','Nb', 'Nd', 'Ne', 'Np', 'Fe', 'B', 'F', 'Sr', 'N', 'Kr', 'Si', 'Sn', 'Sm', 'V', 'Sc', 'Sb', 'Se', 'Co', 'Cl', 'Ca','Ce', 'Xe', 'Tm', 'Cs', 'Cr', 'Cu', 'La', 'Li', 'Tl', 'Lu', 'Th', 'Ti', 'Te', 'Tb', 'Tc', 'Ta', 'Yb', 'Dy', 'I','U', 'Y', 'Ac', 'Ag', 'Ir', 'Al', 'As', 'Ar', 'Au', 'In', 'Mo'] #89

#	all_atomlist = list(set(cod_atomlist+mp_atomlist))

  # You can specify your own element lists (if you don't want to use the above list)
	all_atomlist = ['V','O']


	cod_atomindex = {}
	for i,symbol in enumerate(all_atomlist):
		cod_atomindex[symbol] = i
	return cod_atomlist,cod_atomindex


def get_scale(sigma):
	scale = 1.0/(2*sigma**2)
	return scale

def get_image_one_atom(atom,fakeatoms_grid,nbins,scale):
	grid_copy = fakeatoms_grid.copy()
	ngrid = len(grid_copy)
	image = numpy.zeros((1,nbins**3))
	grid_copy.append(atom)
	drijk = grid_copy.get_distances(-1,range(0,nbins**3),mic=True)
	pijk = numpy.exp(-scale*drijk**2)
	image[:,:] = pijk.flatten()
	return image.reshape(nbins,nbins,nbins)
				
def get_fakeatoms_grid(atoms,nbins):
	atomss = []
	scaled_positions = []
	ijks = []
	grid = numpy.array([float(i)/float(nbins) for i in range(nbins)])
	yv,xv,zv = numpy.meshgrid(grid,grid,grid)
	pos = numpy.zeros((nbins**3,3))
	pos[:,0] = xv.flatten()
	pos[:,1] = yv.flatten()
	pos[:,2] = zv.flatten()
	atomss = Atoms('H'+str(nbins**3))
	atomss.set_cell(atoms.get_cell())#making pseudo-crystal containing H positioned at pre-defined fractional coordinate
	atomss.set_pbc(True)
	atomss.set_scaled_positions(pos)
	fakeatoms_grid = atomss
	return fakeatoms_grid

def get_image_all_atoms(atoms,nbins,scale,norm,num_cores):
	fakeatoms_grid = get_fakeatoms_grid(atoms,nbins)
	cell = atoms.get_cell()
	imageall_gen = Parallel(n_jobs=num_cores)(delayed(get_image_one_atom)(atom,fakeatoms_grid,nbins,scale) for atom in atoms)
	imageall_list = list(imageall_gen)
	cod_atomlist,cod_atomindex = get_atomlist_atomindex()
	nchannel = len(cod_atomlist)
	channellist = []
	for i,atom in enumerate(atoms):
		channel = cod_atomindex[atom.symbol]
		channellist.append(channel)
	channellist = list(set(channellist))
	nc = len(channellist)
	shape = (nbins,nbins,nbins,nc)
	image = numpy.zeros(shape)
	for i,atom in enumerate(atoms):
		nnc = channellist.index(cod_atomindex[atom.symbol])
		img_i = imageall_list[i]
		image[:,:,:,nnc] += img_i * (img_i>=0.02)
		 
	return image,channellist
	
def image2pickle(image,channellist,savefilename):
	dic = {'image':image.tolist(),'channel':channellist.tolist()}
	with open(savefilename,'w') as f:
		json.dump(dic,f)


def basis_translate(atoms):
  N = len(atoms)
  pos = atoms.positions
  cg = np.mean(pos,0)
  dr = 7.5 - cg #move to center of 15A-cubic box
  dpos = np.repeat(dr.reshape(1,3),N,0)
  new_pos = dpos + pos
  atoms_ = atoms.copy()
  atoms_.cell = 15.0*np.identity(3)
  atoms_.positions = new_pos
  return atoms_

def file2image(args):
	inputfiles = args.input_file
	random.shuffle(inputfiles)
	for inputfile in inputfiles:
		tmp = inputfile.split('/')[-1].split('.')[0]; tmp2 = inputfile.split('.')[0]
		touchfile = tmp2+'.touchtouch'
		filename2 = './'+tmp+'_64.pickle'
		savefilename = './'+tmp+'_32.npy'#'.pickle'
		if os.path.isfile(filename2) or os.path.isfile(savefilename):
			print('already made pickle')
			pass
		if os.path.isfile(touchfile):
			pass
		os.system('touch '+touchfile)
		try:
			atoms = read(inputfile,format = args.filetype)
		except:
			os.system('rm '+inputfile)
			continue
		scale = get_scale(sigma=0.26) # values for Gaussain width
		num_cores = args.nproc
		nbins = args.nbins # the number of grid for generated output image
	
		image,channellist = get_image_all_atoms(basis_translate(atoms),nbins,scale,norm,num_cores)
		image2pickle(image,channellist,savefilename)
		os.system('rm '+touchfile)
	return 1

def main():
	parser = argparse.ArgumentParser(description='mapping POSCAR or cif structure into a box image')
	parser.add_argument('--input_file', type=str,nargs='+',
                        help='a file path with the poscar or cif structure')
	parser.add_argument('--filetype', type=str,default='cif',
                        help='filetype : cif,vasp')
	parser.add_argument('--nbins', type=int,default=32,
			help='number of bins in one dimension')
	parser.add_argument('--nproc', type=int,default=1,
			help='number of process')
			
	args = parser.parse_args() 
	file2image(args)
	return 1

if __name__=='__main__':
	main()
	

