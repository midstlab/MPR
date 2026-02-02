⚠️ **Important notice for Bio-protocol readers**
The Bio-protocol article was originally linked to this repository. The correct and up-to-date implementation is available at:

**https://github.com/midstlab/MPR_Bio-Protocol**

If you arrived here via the Bio-protocol article, please use the repository above.


**Codes for "Multiply Perturbed Response to Disclose Allosteric Control of Conformational Change: Application to Fluorescent Biosensor Design"**

**SparseSolution.py:** Code used for the calculation of Omax residues in a given apo-holo pair. Example code here calculates overlaps from **_k_**=1 to **_k_**=5 for both enumeration and optimization approaches.  Solutions are written to `MPR_output.txt`. To use optimization, a Gurobi license should be generated and downloaded.

**DiffE_and_hessian.py:** For superimposition of two structures with Biopython and calculation of inverse hessian with Prody.

**Vector_processing.py:** For visualization of force vectors on ChimeraX 1.8. This code generates a `.bild` file which can be directly launched on ChimeraX. Firstly, it calculates the final coordinates by adding the weights obtained from MPR solution to initial coordinates. Then it creates an arrow between the two points. Arrows can be scaled and customized using the scaling factors.

