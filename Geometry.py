
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from spyrmsd import rmsd as srmsd
# import math

def rotx(deg):
    rad = deg / 180 * np.pi
    return np.array([[1, 0, 0], [0, np.cos(rad), np.sin(rad)], [0, -np.sin(rad), np.cos(rad)]])
def roty(deg):
    rad = deg / 180 * np.pi
    return np.array([[np.cos(rad), 0, -np.sin(rad)], [0, 1, 0], [np.sin(rad), 0, np.cos(rad)]])
def rotz(deg):
    rad = deg / 180 * np.pi
    return np.array([[np.cos(rad), np.sin(rad), 0], [-np.sin(rad), np.cos(rad), 0], [0, 0, 1]])

def make_rot(theta, phi):
    # theta is [0, pi], phi is [0, 2*pi]
    return roty(theta) @ rotz(phi)

def make_rot_xyz(degx, degy, degz):
    return rotx(degx) @ roty(degy) @ rotz(degz)

def rotation_sampling(sampling=18):
    rmat = []
    for i in range(0, 180, 180//sampling):
        for j in range(0, 360, 360//sampling):
            #for k in range(0, 360, 360//sampling):
                rmat.append(roty(i) @ rotz(j))
    rmat = np.array(rmat)
    return rmat

def raster_unit_sphere(num=200):
    L = np.sqrt(num * np.pi);
    pt = []
    for i in range(num):
        h = 1.0 - (2.0 * i + 1.0) / num
        p = np.arccos(h)
        t = L * p
        xu = np.sin(p) * np.cos(t)
        yu = np.sin(p) * np.sin(t)
        zu = np.cos(p)
        pt.append([xu, yu, zu])

    return np.array(pt) 

def Kabsch_RMSD(P, Q):
    return ALIGN_A_RMSD_B(P, Q)

# Kabsch align two point groups
def ALIGN_A_RMSD_B(P, Q, A=None, B=None):
    # P is the one to be aligned (N * 3)
    # Q is the ref (N * 3)
    # A is the list of index to be considered for alignment (protein) (N * 1)
    # B is the list of index to calculate RMSD (ligand) (N * 1)
    # Returns rmsd between subset P[B] and Q[B]
    if A is not None:
        PU = P[A] # Get subset
        QU = Q[A] # Get subset
    else:
        PU = P
        QU = Q
    PC = PU - PU.mean(axis=0) # Center points
    QC = QU - QU.mean(axis=0) # Center points
    # Kabsch method
    C = np.dot(np.transpose(PC), QC)
    V, S, W = np.linalg.svd(C,full_matrices=False)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    # Create Rotation matrix U
    U = np.dot(V, W)
    P = P - PU.mean(axis=0) # Move all points
    Q = Q - QU.mean(axis=0) # Move all points
    P = np.dot(P, U) # Rotate P
    if B is not None:
        diff = P[B] - Q[B]
        N = len(P[B])
    else:
        diff = P - Q
        N = len(P)
    return np.sqrt((diff * diff).sum() / N), P + QU.mean(axis=0) 

def pureRMSD(P, Q):
    # Assume P and Q are aligned first
    diff = P - Q
    N = len(P)
    return np.sqrt((diff * diff).sum() / N)

def symmRMSD(P, Q, atom_prop, adj_mtx, isomorphism):
    # Symmetry corrected RMSD
    return cached_symmrmsd(P, Q, atom_prop, atom_prop, adj_mtx, adj_mtx, isomorphisms=isomorphism)

def cached_symmrmsd(
    coordsref: np.ndarray,
    coords: Union[np.ndarray, List[np.ndarray]],
    apropsref: np.ndarray,
    aprops: np.ndarray,
    amref: np.ndarray,
    am: np.ndarray,
    center: bool = False,
    minimize: bool = False,
    cache: bool = True,
    atol: float = 1e-9,
    isomorphisms = None
) -> Any:
    """
    Compute RMSD using graph isomorphism for multiple coordinates.

    Parameters
    ----------
    coordsref: np.ndarray
        Coordinate of reference molecule
    coords: List[np.ndarray]
        Coordinates of other molecule
    apropsref: np.ndarray
        Atomic properties for reference
    aprops: np.ndarray
        Atomic properties for other molecule
    amref: np.ndarray
        Adjacency matrix for reference molecule
    am: np.ndarray
        Adjacency matrix for other molecule
    center: bool
        Centering flag
    minimize: bool
        Minimum RMSD
    cache: bool
        Cache graph isomorphisms
    atol: float
        Absolute tolerance parameter for QCP (see :func:`qcp_rmsd`)

    Returns
    -------
    float: Union[float, List[float]]
        Symmetry-corrected RMSD(s) and graph isomorphisms

    Notes
    -----

    Graph isomorphism is introduced for symmetry corrections. However, it is also
    useful when two molecules do not have the atoms in the same order since atom
    matching according to atomic numbers and the molecular connectivity is
    performed. If atoms are in the same order and there is no symmetry, use the
    `rmsd` function.
    """

    RMSD, isomorphism = srmsd._rmsd_isomorphic_core(
        coordsref,
        coords,
        apropsref,
        aprops,
        amref,
        am,
        center=center,
        minimize=minimize,
        isomorphisms=isomorphisms,
        atol=atol,
    )

    return RMSD, isomorphism


def rotate_by_axis(v, a0, a1, degrees):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians. Using the Euler-Rodrigues formula:
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = np.asarray(a1-a0)
    axis = axis / np.sqrt(np.dot(axis, axis))
    theta = degrees * np.pi / 180
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return np.dot(rot_mat, (v-a1).T).T + a1

def rotate_by_3pt(pt, degrees, v0, v1=None):
#     print('v0', v0, '\nv1', v1, '\npt', pt, '\ndeg', degrees)
    pv0 = pt[0] - pt[1]
    pv1 = pt[2] - pt[1]
    axis = np.cross(pv0, pv1)
    axis = axis / np.sqrt(np.dot(axis, axis))
    #theta = -degrees * np.pi / 180 / 2.0 # Rotate half
    theta = -degrees * np.pi / 180 # Rotate full
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    v0 = np.dot(rot_mat, (v0 - pt[1]).T).T + pt[1]
    #theta *= -1
    #b, c, d = -axis * np.sin(theta / 2.0)
    #aa, bb, cc, dd = a * a, b * b, c * c, d * d
    #bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    #rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
    #                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
    #                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    #v1 = np.dot(rot_mat, (v1 - pt[1]).T).T + pt[1]

    return v0#, v1
    
def rotate_then_center(lig_coord, rot, xyz=np.array([0,0,0])):
    return (rot @ (lig_coord - lig_coord.mean(0)).T).T + xyz
