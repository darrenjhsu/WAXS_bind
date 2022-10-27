
import numpy as np
from scipy.spatial import distance_matrix
from scipy.ndimage import generate_binary_structure, binary_dilation, convolve
from saxstats import saxstats
import time
from xs_helper import xray_scatter 
from array import array
from Geometry import *

def sok(radius = 5.0, grid_spacing = 0.5, near_radius = 1.5):
    print(f'Creating a spherical occlusion kernel of radius {radius:.3f} A on a grid with {grid_spacing:.3f} A spacing')
    nx_half = np.around(radius / grid_spacing).astype(int)
    nx = 2 * nx_half + 1
    x_ = np.linspace(-grid_spacing * nx_half, grid_spacing * nx_half, nx)
    v = np.zeros((nx, nx, nx)).flatten()
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    r = np.sqrt(x**2 + y**2 + z**2)
    rbins = np.linspace(0, radius, nx_half + 1) # Using radius as the shell
    rbin_labels = np.searchsorted(rbins, r.flatten())
    for i in np.unique(rbin_labels):
        v[rbin_labels==i] = 1 # / np.sum(rbin_labels == i)
    v = v.reshape(nx, nx, nx)
    v[r > radius] = 0        # Discard anything beyond radius
    v[r < near_radius] = 0        # Discard anything beyond radius
    v /= np.sum(v)           # Normalize kernel so when convolving the value is conserved
    return v

def min_max(xyz, margin=0.0, spacing=0.2, align=False):
    if align:
        minxyz = np.floor((np.min(xyz,axis=-2) - margin) / spacing) * spacing
        maxxyz = np.ceil((np.max(xyz,axis=-2) + margin) / spacing) * spacing
        return minxyz, maxxyz
    return np.min(xyz, axis=-2)-margin, np.max(xyz, axis=-2)+margin

def grid(xyz, spacing=0.2, margin=1.0, make_cubic=False, center_average=False):
    avg_xyz = np.mean(xyz, axis=0)
    min_xyz, max_xyz = min_max(xyz, margin=margin, spacing=spacing, align=True)
    #print(min_xyz, max_xyz, max_xyz-min_xyz)
    if make_cubic:
        maxd = np.max(max_xyz - min_xyz)
        min_xyz_2 = min_xyz - (maxd - (max_xyz - min_xyz)) / 2.0
        max_xyz_2 = max_xyz + (maxd - (max_xyz - min_xyz)) / 2.0
        min_xyz = min_xyz_2
        max_xyz = max_xyz_2
        #print(min_xyz, max_xyz, max_xyz-min_xyz)
    if center_average:
        maxd = np.vstack([max_xyz - avg_xyz, avg_xyz - min_xyz]).max(0)
        #print(avg_xyz, maxd)
        min_xyz = avg_xyz - maxd
        max_xyz = avg_xyz + maxd
        #print(min_xyz, max_xyz, max_xyz-min_xyz)
    minx, miny, minz = min_xyz
    maxx, maxy, maxz = max_xyz
    sxyz = max_xyz - min_xyz
    nxyz = np.ceil((max_xyz - min_xyz) / spacing).astype(int) + 1
    pxyz = np.around((xyz - min_xyz) / spacing).astype(int)
    nx, ny, nz = nxyz
    gx_ = np.linspace(minx, maxx, nx)
    gy_ = np.linspace(miny, maxy, ny)
    gz_ = np.linspace(minz, maxz, nz) 
    gx, gy, gz = np.meshgrid(gx_, gy_, gz_, indexing='ij')
    gxyz = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T
    return nxyz, gxyz, pxyz, sxyz, min_xyz, max_xyz

def overlap_grid(protein, ligand, conformerID=0, rotation=None, pocket=None, grid_spacing=0.2, 
                 radius=1.0, write_grid=False, timing=False, printing=True):

    t0 = time.time()
    # Generate a ligand and a protein grid
    if not protein.has_grid or grid_spacing != protein.grid_spacing or protein.radius != radius:
        print('Generate grid data for protein')
        protein.create_grid(grid_spacing, radius)
    protein_nxyz = protein.protein_nxyz
    protein_gxyz = protein.protein_gxyz
    protein_pxyz = protein.protein_pxyz
    protein_sxyz = protein.protein_sxyz
    protein_volume = protein.protein_volume
    num_voxels = np.size(protein_volume)
    protein_near_zone = protein.protein_near_zone
    t1 = time.time()
    if timing:
        print(f'{(t1-t0)*1000:.3f} ms protein grid generation')
    
    #print(protein_nxyz, ligand_nxyz, protein_gxyz.shape, protein_coords.shape, ligand_gxyz.shape, ligand_coords.shape)
    ligand_coords = ligand.get_coordinates(conformerID)[ligand.elements != 'H']
    if rotation is not None: # It then should be 3x3 rotational matrix
        ligand_coords = ligand_coords @ rotation
    ligand_nxyz, ligand_gxyz, ligand_pxyz, ligand_sxyz, _, _ = grid(ligand_coords, grid_spacing, margin=radius, make_cubic=False, center_average=True)
    ligand_volume = np.any(distance_matrix(ligand_gxyz, ligand_coords) - radius < 0, axis=-1)
    ligand_volume = ligand_volume.reshape(ligand_nxyz)
    #print("Done with ligand volume")
    t2 = time.time()
    if timing:
        print(f'{(t2-t1)*1000:.3f} ms ligand grid generation')

    #print(np.sum(protein_exclude_zone))
    #protein_volume = ~binary_dilation(protein_volume, structure=np.flip(ligand_volume))
    t3 = time.time()
    if timing:
        print(f'{(t3-t2)*1000:.3f} ms binary dilation')

    #protein_volume *= protein_near_zone
    protein_volume = np.ones_like(protein_volume)
    if printing:
        print(f'There are {np.sum(protein_volume)} points deemed possible out of {num_voxels} points purely from overlap.')
    t4 = time.time()
    if timing:
        print(f'{(t4-t3)*1000:.3f} ms remove far zone')

    if pocket is not None: # pocket is a PDB object
        if not pocket.has_grid:
            pocket.create_grid(grid_spacing, 3.0, existing_grid=protein, element='Ve')
        pocket_volume = pocket.protein_volume
        #pv2 = distance_matrix(protein_gxyz[protein_volume.flatten()], pocket_coord).min(1) > 2
        protein_volume *= pocket_volume
        t41 = time.time()
        if timing:
            print(f'{(t41-t4)*1000:.3f} ms pocket distance calculation')
        #pv2_index = np.arange(num_voxels)[protein_volume.flatten()][pv2]
        #protein_volume.ravel()[pv2_index] = False
        #t42 = time.time()
        #if timing:
        #    print(f'{(t42-t41)*1000:.3f} ms matrix assignment')
        if printing:
            print(f'There are {np.sum(protein_volume)} points deemed possible out of {num_voxels} points after pocket overlap.')
    t5 = time.time()
    if timing:
        print(f'{(t5-t4)*1000:.3f} ms pocket filter')

    if write_grid:
        write_grid_points(protein_gxyz, protein_volume.flatten(), 'grid2.pdb')
    t6 = time.time()
    if timing:
        print(f'{(t6-t5)*1000:.3f} ms write out grid points')

    return protein_volume, protein_gxyz, protein_sxyz, ligand_volume, ligand_gxyz, ligand_sxyz


 
def write_pdb_line(f,*j,endline=False):
    j = list(j)
    j[0] = j[0].ljust(6)#atom#6s
    j[1] = j[1].rjust(5)#aomnum#5d
    j[2] = j[2].center(4)#atomname$#4s
    j[3] = j[3].ljust(3)#resname#1s
    j[4] = j[4].rjust(1) #Astring
    j[5] = j[5].rjust(4) #resnum
    if not endline:
        j[6] = str('%8.3f' % (float(j[6]))).rjust(8) #x
        j[7] = str('%8.3f' % (float(j[7]))).rjust(8)#y
        j[8] = str('%8.3f' % (float(j[8]))).rjust(8) #z
        j[9] =str('%6.2f'%(float(j[9]))).rjust(6)#occ
        j[10]=str('%6.2f'%(float(j[10]))).ljust(6)#temp
        j[11]=j[11].rjust(12)#elname
    else:
        j[6] = ''.rjust(8)
        j[7] = ''.rjust(8)
        j[8] = ''.rjust(8)
        j[9] = ''.rjust(6)
        j[10] = ''.ljust(6)
        j[11] = ''.rjust(12)
    f.write("%s%s %s %s %s%s    %s%s%s%s%s%s\n"% (j[0],j[1],j[2],j[3],j[4],j[5],j[6],j[7],j[8],j[9],j[10],j[11]))


def write_grid_points(xyz, v, fname='grid.pdb', O_col=None, B_col=None):
    xyzob = np.zeros((xyz.shape[0], xyz.shape[1]+2))
    xyzob[:,:3] = xyz
    if O_col is not None:
        assert O_col.shape == v.shape
        xyzob[:,3] = O_col
    if B_col is not None:
        assert B_col.shape == v.shape
        xyzob[:,4] = B_col
    with open(fname,'w') as f:
        for idx, ijkob in enumerate(xyzob[v]):
            write_pdb_line(f,'ATOM', str(idx+1), 'X', 'XXX', 'A', str(1), *ijkob, 'X')

class Scatter:
    def __init__(self, q=np.linspace(0, 1, 200), c1=1.0, c2=2.0, r_m=1.62, sol_s=1.8, num_raster=512, rho=0.334):
        self.q = q
        self.c1 = c1
        self.c2 = c2
        self.r_m = r_m
        self.sol_s = sol_s
        self.num_raster = num_raster
        self.rho = rho

    def scatter(self, protein=None, ligand=None, ligand_coords=None):
        if protein is None and ligand is None:
            print("No input, return None")
            return None
        # prepare coordinates from protein and optionally ligand
        coords = np.empty((0, 3))
        ele = np.empty(0)
        if protein is not None:
            coords = protein.pdb.coords
            ele = np.concatenate([ele, protein.electrons - 1])
        if ligand is not None:
            # do things with ligand
            if ligand_coords is not None:
                if len(ligand_coords) == ligand.num_atoms:
                    coords = np.vstack([coords, ligand_coords])
                    ele = np.concatenate([ele, ligand.electrons - 1])
                elif len(ligand_coords) == np.sum([x != 'H' for x in ligand.elements]):
                    coords = np.vstack([coords, ligand_coords])
                    ele = np.concatenate([ele, ligand.electrons[ligand.electrons != 'H'] - 1])
                else:
                    raise ValueError("ligand_coords coordinates must have ligand.num_atom atoms or that minus hydrogens")
            else:
                coords = np.vstack([coords, ligand.get_coordinates()])
                ele = np.concatenate([ele, ligand.electrons - 1])
                    
        
        # array-ize np arrays
        coords_a = array('f', coords.flatten())
        ele_a = array('I', ele.astype(int))
        q_a = array('f', self.q)

        t0 = time.time()
        S_calc = xray_scatter(coords_a, ele_a, q_a, 
                              num_raster=self.num_raster, sol_s=self.sol_s, 
                              r_m=self.r_m, rho=self.rho, c1=self.c1, c2=self.c2)
        t1 = time.time()

        #print(f'C = {(t1-t0)*1000:.3f} ms')

        return S_calc

