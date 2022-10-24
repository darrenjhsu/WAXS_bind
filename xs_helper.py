import ctypes
from ctypes import *
import sys

lib_path = './bin/XSMD.so' 

try:
    xs = CDLL(lib_path)
    print('XS loaded successfully')
except Exception as e:
    print('CDLL failed')
    print(e)

xs_calc = xs.xray_scattering
xs_calc.restype = None

def xray_scatter(coords, ele, q, 
                 num_raster=512, sol_s=1.8, r_m=1.62, rho=0.334, c1=1.00, c2=2.00):
    
    assert len(coords) == 3 * len(ele)
    num_atom = len(ele)
    num_coords = 3 * num_atom
    num_q = len(q)
    print(f'num atom: {num_atom}, num_coords: {num_coords}, num q: {num_q}')
    c_coords = (c_float * num_coords)(*coords)
    c_ele = (c_int * num_atom)(*ele)
    c_q = (c_float * num_q)(*q)
    c_S_calc = (c_float * num_q)()

    xs_calc(c_int(num_atom), c_coords, c_ele, c_int(num_q), c_q, c_S_calc, 
            c_int(num_raster), c_float(sol_s), c_float(r_m), c_float(rho), c_float(c1), c_float(c2))
    return c_S_calc[:]
