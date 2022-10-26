
import numpy as np


def rotx(deg):
    rad = deg / 180 * np.pi
    return np.array([[1, 0, 0], [0, np.cos(rad), np.sin(rad)], [0, -np.sin(rad), np.cos(rad)]])

def roty(deg):
    rad = deg / 180 * np.pi
    return np.array([[np.cos(rad), 0, -np.sin(rad)], [0, 1, 0], [np.sin(rad), 0, np.cos(rad)]])
def rotz(deg):
    rad = deg / 180 * np.pi
    return np.array([[np.cos(rad), np.sin(rad), 0], [-np.sin(rad), np.cos(rad), 0], [0, 0, 1]])

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
