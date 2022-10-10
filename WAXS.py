
import numpy as np
from scipy.spatial import distance_matrix
from scipy.ndimage import generate_binary_structure, binary_dilation, convolve
from saxstats import saxstats

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
    print(min_xyz, max_xyz, max_xyz-min_xyz)
    if make_cubic:
        maxd = np.max(max_xyz - min_xyz)
        min_xyz_2 = min_xyz - (maxd - (max_xyz - min_xyz)) / 2.0
        max_xyz_2 = max_xyz + (maxd - (max_xyz - min_xyz)) / 2.0
        min_xyz = min_xyz_2
        max_xyz = max_xyz_2
        print(min_xyz, max_xyz, max_xyz-min_xyz)
    if center_average:
        maxd = np.vstack([max_xyz - avg_xyz, avg_xyz - min_xyz]).max(0)
        print(avg_xyz, maxd)
        min_xyz = avg_xyz - maxd
        max_xyz = avg_xyz + maxd
        print(min_xyz, max_xyz, max_xyz-min_xyz)
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
    return nxyz, gxyz, pxyz, sxyz

def overlap_grid(protein, ligand, conformerID=0, grid_spacing=0.2, radius=1.0):
    # Generate a ligand and a protein grid
    protein_coords = protein.pdb.coords
    ligand_coords = ligand.get_coordinates(conformerID)
    ligand_nxyz, ligand_gxyz, ligand_pxyz, ligand_sxyz = grid(ligand_coords, grid_spacing, margin=radius, make_cubic=False, center_average=True)
    protein_nxyz, protein_gxyz, protein_pxyz, protein_sxyz = grid(protein_coords, grid_spacing, margin=radius*3)
    print(protein_nxyz, ligand_nxyz, protein_gxyz.shape, protein_coords.shape, ligand_gxyz.shape, ligand_coords.shape)
    ligand_volume = np.any(distance_matrix(ligand_gxyz, ligand_coords) - radius < 0, axis=-1)
    ligand_volume = ligand_volume.reshape(ligand_nxyz)
    print("Done with ligand volume")
    protein_volume = np.zeros(protein_nxyz).astype(bool)
    protein_volume[protein_pxyz[:,0], protein_pxyz[:,1], protein_pxyz[:,2]] = True
    write_grid_points(protein_gxyz, protein_volume.flatten())
    print(np.sum(protein_volume))
    near_dilation = np.around((protein.near_radius+3) / grid_spacing).astype(int)
    protein_near_zone = binary_dilation(protein_volume, iterations=near_dilation)
    print(np.sum(protein_near_zone))
    exclude_dilation = np.around(radius / grid_spacing).astype(int)
    protein_exclude_zone = ~binary_dilation(protein_volume, iterations=exclude_dilation)
    protein_volume = ~protein_exclude_zone
    print(np.sum(protein_exclude_zone))
    #sok_kernel = sok(grid_spacing=grid_spacing)
    #so_factor = convolve(protein_volume.astype(float), sok_kernel, mode='constant')
    #print(so_factor.min(), so_factor.max())
    ##print(np.around(ligand_nxyz/2).astype(int))
    #protein_volume = ~convolve(protein_volume, ligand_volume)
    protein_volume = ~binary_dilation(protein_volume, structure=np.flip(ligand_volume))
    protein_volume *= protein_near_zone
    #protein_volume *= protein_exclude_zone
    #                                 #origin=tuple(-(x+1)//2 +1  for x in ligand_nxyz))
    #                                 #origin=(14, 14, 14))
    print(np.sum(protein_volume))
    #write_grid_points(protein_gxyz, protein_volume.flatten(), 'grid2.pdb', O_col=so_factor.flatten()*100)
    write_grid_points(protein_gxyz, protein_volume.flatten(), 'grid2.pdb')

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




