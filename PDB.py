
import numpy as np
from saxstats import saxstats

class PDB:
    def __init__(self, fname):
        self.fname = fname
        self.pdb = saxstats.PDB(fname)
        self.elements = self.pdb.atomtype
        self.grid = None
        self.hasManyH = np.sum([x == 'H' for x in self.elements]) / np.sum([x != 'H' for x in self.elements]) > 0.5
        self.near_radius = 4.0
        if self.hasManyH:
            print('This PDB seem to have hydrogens modeled')
        else:
            print('This PDB does not seem to have hydrogens modeled')
