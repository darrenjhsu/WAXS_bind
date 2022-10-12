
import numpy as np
from saxstats import saxstats
from WAXS import *

class PDB:
    def __init__(self, fname):
        self.fname = fname
        self.pdb = saxstats.PDB(fname)
        self.elements = self.pdb.atomtype
        self.has_grid = None
        self.hasManyH = np.sum([x == 'H' for x in self.elements]) / np.sum([x != 'H' for x in self.elements]) > 0.5
        self.near_radius = 8.0
        self.protein_nxyz = None # Number of points each dimension
        self.protein_gxyz = None # Grid xyz (N_size of protein_nxyz, 3)
        self.protein_pxyz = None # Where each atom is at the grid point
        self.protien_sxyz = None # Side length of protein grid
        self.grid_spacing = None
        self.radius = None
        if self.hasManyH:
            print('This PDB seem to have hydrogens modeled')
        else:
            print('This PDB does not seem to have hydrogens modeled')

    def create_grid(self, grid_spacing, radius, existing_grid=None, element=None):
        if existing_grid is not None:
            assert existing_grid.has_grid, "PDB object for existing_grid must first have a grid"
        # existing grid is actually a PDB object
        if element is not None:
            coords = self.pdb.coords[self.elements == 'Ve']
        else:
            coords = self.pdb.coords
        if existing_grid is not None:
            self.grid_spacing = existing_grid.grid_spacing
        else:
            self.grid_spacing = grid_spacing
        self.radius = radius
        if existing_grid is not None:
            protein_nxyz = existing_grid.protein_nxyz
            protein_gxyz = existing_grid.protein_gxyz
            protein_sxyz = existing_grid.protein_sxyz
            min_xyz = existing_grid.min_xyz
            max_xyz = existing_grid.max_xyz
            protein_pxyz = np.around((coords - min_xyz) / self.grid_spacing).astype(int)
            # use that grid (protein_n/g/sxyz in particular)
            # calculate own protein_pxyz
        else:
            protein_nxyz, protein_gxyz, protein_pxyz, protein_sxyz, min_xyz, max_xyz = grid(coords, grid_spacing, margin=radius*3)
        protein_volume = np.zeros(protein_nxyz).astype(bool)
        num_voxels = np.size(protein_volume)
        protein_volume[protein_pxyz[:,0], protein_pxyz[:,1], protein_pxyz[:,2]] = True
        near_dilation = np.around((self.near_radius) / grid_spacing).astype(int)
        protein_near_zone = binary_dilation(protein_volume, iterations=near_dilation)
        exclude_dilation = np.around(radius / grid_spacing).astype(int)
        protein_volume = binary_dilation(protein_volume, iterations=exclude_dilation)

        self.protein_nxyz = protein_nxyz
        self.protein_gxyz = protein_gxyz
        self.protein_pxyz = protein_pxyz
        self.protein_sxyz = protein_sxyz
        self.protein_volume = protein_volume # Possible locations 
        self.protein_near_zone = protein_near_zone # Some radius around the protein atoms
        self.min_xyz = min_xyz
        self.max_xyz = max_xyz
        self.has_grid = True



