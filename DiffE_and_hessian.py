from prody import *
import numpy as np
import spcoords
from math import log10, floor, sqrt
def round_sig(x, sig=2):
    return round(x,sig-int(floor(log10(x)))-1)

# Superimpose final_holo onto initial_apo structure and calculate displacement (DiffE)

pdb = parsePDB("final_holo.pdb")
structure_calpha = pdb.select("calpha")
initial_apo = parsePDB("initial_apo.pdb").select("calpha").getCoords()
final_holo = parsePDB("final_holo.pdb").select("calpha").getCoords()
residue_number = len(structure_calpha)
final_alg = spcoords.superposer(initial_apo, final_holo)
diffE = (final_alg[0]-initial_apo).reshape(residue_number*3, 1)
np.savetxt('diffE.dat', diffE)
rms = sup.get_rms()


#Calculate hessian for initial_apo

initial_apo_anm = ANM('')
initial_apo_anm.buildHessian(initial_apo, cutoff=12, gamma=1)
initial_apo_hes = initial_apo_anm.getHessian()
print(initial_apo_anm.getEigvals())
initial_apo_inv_hes = np.linalg.pinv(initial_apo_hes)
np.savetxt("apo_hessian.dat", initial_apo_hes)
np.savetxt("apo_inv_hessian.dat", initial_apo_inv_hes)







