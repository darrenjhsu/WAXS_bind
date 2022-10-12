
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

def rotation_sampling(sampling=8):
    rmat = []
    for i in range(0, 360, 360//sampling):
        for j in range(0, 360, 360//sampling):
            for k in range(0, 360, 360//sampling):
                rmat.append(rotx(i) @ roty(j) @ rotz(k))
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

