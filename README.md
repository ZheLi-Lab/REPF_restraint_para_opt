# REPF_Parameters-Optimization 
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
pip install mdtraj==1.9.7 
pip install parmed==2.11.0
pip install scipy==1.9.1
## The REPF_parameters optimization procedure
- It was shown in detail in the REPF_Parameters-Optimization.ipynb
## Reference
