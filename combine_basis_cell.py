import numpy as np
import glob

from tqdm import tqdm

from ase.io import read,write

for name in glob.glob('./../cell/*.cif'):
  cif_name = name.split('/')[-1]
  print cif_name
  cell = read(name)
  cg_cell = cell.get_positions()

  real_mat = read('./../basis/'+cif_name)
  pos = real_mat.get_positions()

  delta = cg_cell - np.mean(pos,0)
  new_pos = pos+delta

  real_mat.set_cell(cell.get_cell())
  real_mat.set_positions(new_pos)

  write('./'+cif_name[:-4]+'.vasp',real_mat)
