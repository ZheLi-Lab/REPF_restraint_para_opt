# REPF_Parameters-Optimization 
The REPF optimization parameters are based on the PLUMED plugin. Please ensure that you provide the corresponding files. Analysis is performed by providing the appropriate plumed input file (plumed.dat) and the corresponding plumed output data (Colvar).

## Detailed Description of REPF_Parameters-Optimization
The following is a detailed explanation of the relevant files for optimizing the restrained degrees of freedom in REPF:

- Preparation of Input Files
```sh
By default, the plumed input file is plumed.dat.
This file contains the indices of the atoms involved in forming the six restrained degrees of freedom for the system, including one distance, two angles, and three dihedral angles.

The plumed output file is Colvar.
 This file contains numerical records of the distances, angles, and dihedral parameters of the restraint atoms.
```
- Output of the Restrained Scheme
```sh
After optimization, a file named res_databystd.csv will be generated.
```
- The following is an example and explanation of the res_databystd.csv file:
```sh
,"restraint_atom[lig1, lig2, lig3, rec1, rec2, rec3](start from 1)","rec_atoms[rec3, rec2, rec1](start from 0)","lig_atoms[lig1, lig2, lig3](start from 0)",distance between lig1 and rec1(unit: Angstrom),angle between lig1 rec1 and rec2(unit:radian),angle between lig2 lig1 and rec1(unit:radian),dihedral between lig1 rec1 rec2 and rec3(unit:radian),dihedral between lig2 lig1 rec1 and rec2(unit:radian),dihedral between lig3 lig2 lig1 and rec1(unit:radian),cal_ene(unit:kcal/mol),cost function value
0,"[3358, 3357, 3355, 660, 667, 669]","[668, 666, 659]","[3357, 3356, 3354]",5.797172466666666,1.9593565942666669,0.9378565543999999,-2.219771983066668,2.8428033988810277,0.5921863020000004,0.3583226812216338,0.014669222749200333

It includes three restraint atoms from the ligand (lig1, lig2, lig3) and three restraint atoms from the complex system (rec3, rec2, rec1), as well as the values of the six restrained degrees of freedom.
```
## Software Installation 
- Create conda environment
```sh 
conda create --name openmm-plumed
conda activate openmm-plumed
conda install openmm cudatoolkit=11.0 -c conda-forge
conda install openmmtools plumed openmm-plumed tqdm -c conda-forge
```
- Additional required package
```sh 
pip install numpy==1.22.4
pip install pandas==1.4.4
pip install matplotlib==3.6.0

```
## The REPF_parameters optimization procedure
- It was shown in detail in the REPF_Parameters-Optimization.ipynb
## Reference
