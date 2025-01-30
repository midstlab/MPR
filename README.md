**Codes for "Multiply Perturbed Response to Disclose Allosteric Control of Conformational Change: Application to Fluorescent Biosensor Design"**

**SparseSolution.py:** Code used for the calculation of Omax residues in a given apo-holo pair. Example code here calculates overlaps from **_k_**=1 to **_k_**=5 for both enumeration and optimization approaches. To use optimization, a Gurobi license should be generated and downloaded. Solutions are written to `MPR_output.txt`.

**DiffE_and_hessian.py:** For superimposition of two structures with Biopython and calculation of inverse hessian with Prody.

**Vector_processing.py:** For visualization of force vectors on ChimeraX. This code generates a `.bild` file using the initial coordinates and weights obtained from an MPR solution. Force arrows can be scaled and customized using the scaling factors.
