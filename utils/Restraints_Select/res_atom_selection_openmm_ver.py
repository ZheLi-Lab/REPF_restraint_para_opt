from .restraint_aly_tools_openmm_ver import ALL_Restrain_Data, Restrain_Data, RED_FIT, RestraintParam
import numpy as np
import pandas as pd
import ast
import re
import os

class Res_atom_select():
    def __init__(self,plumed_input_file, plumed_output_file):
        '''Initializing
        
        Parameters
        ----------
        plumed_input_file: str
            Name of the plumed input file, which will be generated by this function. 
        plumed_output_file: str
            Name of the plumed output file, which will be generated by the following molecular dynamic simulation.
        plumed_cal_freq: int
            Specifies the frequence with which the collective variables of interest should be output.
        '''
        self.plumed_input_file = plumed_input_file
        self.plumed_output_file = plumed_output_file
    
    def aly_traj_get_best_rest(self, lambda_group, opt_cost_name=None, if_mean=False, if_init_pose=False):
        '''After preliminary MD, analyze the output file of plumed, and obtain the optimal restraint atom group and corresponding restraint parameters based on the RED formula. And output the internal energy difference between the free state and the next constrained state and return a class includes all the information needed for the harmonic restints addition during the alchemical transformation MD.

        Parameters
        ----------
        lambda_group: dict
            This dict is used to specify the lambda scheme.
            Like:  lambdas_group_12stp={
                                        'lambda_restraints'     : [0.00, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00,],
                                        'lambda_electrostatics' : [1.00, 1.00,  1.00, 1.00,  1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,],
                                        'lambda_sterics'        : [1.00, 1.00,  1.00, 1.00,  1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,]
                                        }
        fake_state_xml: str, default: 'state_s0.xml' or None
            The name of fake state xml file that is used to skip the alchemical simulation of the free state (that is the first lambda)
        nsteps: int
            The simulation steps.
        timestep: float, unit: femtoseconds, like: 4 
            The simulation update timestep.

        Return 
        ----------
        res_parm_format: <class 'RestraintParam'>
            A class includes all the information needed for the harmonic restaints addition during the alchemical transformation MD.

        Example
        ----------
        >>> from utils.restraint_aly_tools_openmm_ver import *
        >>> test_ = Res_atom_select('protein.rst7', 'protein.prmtop', 'plumed.dat', 'Colvar', 100)
        >>> test_.get_restr_plumed_input('MOL', True)
        After preliminary MD:
        >>> lambdas_group_12stp={
                                'lambda_restraints'     : [0.00, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00,],
                                'lambda_electrostatics' : [1.00, 1.00,  1.00, 1.00,  1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,],
                                'lambda_sterics'        : [1.00, 1.00,  1.00, 1.00,  1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,]
                                }
        >>> res_parm = test_.aly_traj_get_best_rest(lambdas_group_12stp, 'state_s0.xml')
        '''
        all_data_path = self.plumed_output_file
        res = ALL_Restrain_Data(all_data_path=all_data_path, muti_six_atm_lst=self.muti_six_atm_lst, fraction=0.75)
        self.All_res_obj = res
        res.process_data(opt_cost_name, if_mean, if_init_pose)
        res.generate_csv(csv_name="res_data_ori.csv", lst=res.restrain_data_list)
        res.generate_csv(csv_name="res_databyfopt.csv", lst=res.sort_result())

        count = -1
        for i in res.sort_result():
            count+=1
            if count < 20:
                print('The serial number is {}.'.format(str(count+1)))
                if getattr(self, 'based_lig_shape_res_atom', None) and i.restrain_group in self.based_lig_shape_res_atom:
                    print('Six atoms used for restraints are :' + str(i.restrain_group) + ', they are selected by ligand shape strategy.')
                    print('The cost function value is ' + str(i.fopt))
                elif getattr(self, 'based_HB_pair_res_atom', None) and i.restrain_group in self.based_HB_pair_res_atom:
                    print('Six atoms used for restraints are :' + str(i.restrain_group) + ', they are selected by HBond pair strategy.')
                    print('The cost function value is ' + str(i.fopt))
                elif getattr(self, 'based_HB_mainchain_res_atom', None) and i.restrain_group in self.based_HB_mainchain_res_atom:
                    print('Six atoms used for restraints are :' + str(i.restrain_group) + ', they are selected by HBond mainchain strategy.')
                    print('The cost function value is ' + str(i.fopt))
                elif getattr(self, 'based_Huggins', None) and i.restrain_group in self.based_Huggins:
                    print('Six atoms used for restraints are :' + str(i.restrain_group) + ', they are selected by Huggins strategy.')
                    print('The cost function value is ' + str(i.fopt))
        best=res.get_best_result()
        self.best_res_Data_obj = best
        self.all_res_data = res
        res_parm = best.get_res_parm_4_openmm()
        res_parm_format = RestraintParam(rec_atoms=res_parm.rec_atoms,lig_atoms=res_parm.lig_atoms,r=res_parm.r,theta1=res_parm.theta1,theta2=res_parm.theta2,phi1=res_parm.phi1,phi2=res_parm.phi2,phi3=res_parm.phi3)
        
        output_dU = best.delta_u
        #self.gen_mbarlike_resene_csv(output_dU, lambda_group, nsteps, timestep, self.plumed_cal_freq, first_state_csv)

        res.generate_csv(csv_name="res_databystd.csv", lst=res.res_list_4_std_sort)
        best.draw_figure('best.png')
        with open('restr.txt', 'w') as restr_file:
            restr_file.write(best.echo_rest_txt('openmm'))


        return res_parm_format
    def defi_rest_atoms(self,plumed_input_file):
        '''plumed_input_input:the plumed input file, which will be readed by this function;default:plumed.dat.
        r_0: DISTANCE ATOMS=lig1, rec1
        thetaA_0: ANGLE ATOMS=lig1, rec1, rec2
        thetaB_0: ANGLE ATOMS=lig2, lig1, rec1
        phiA_0: TORSION ATOMS=rec3, rec2, rec1, lig1
        phiB_0: TORSION ATOMS=rec2, rec1, lig1, lig2
        phiC_0: TORSION ATOMS=rec1, lig1, lig2, lig3
        PRINT ARG=r_0,thetaA_0,thetaB_0,phiA_0,phiB_0,phiC_0, FILE=plumed_output_file STRIDE=output_frq
        FLUSH STRIDE=output_frq
        '''
       
        atom_list = []
        res=[]
        lig=[]
        
        with open(plumed_input_file, 'r') as file:
            lines = file.readlines()
            filtered_lines = lines[:-2] 
            
            for line in filtered_lines:
                                
                if 'thetaA_0' in line:   
                    parts = line.split('=')   
                    if len(parts) > 1:
                        atoms_part = parts[1]
                        atoms = [atom.strip() for atom in atoms_part.split(',')]
                        atoms = atoms[1:]
                        #print(atoms)
                        res = [int(num) for num in atoms]
                        #print(res)
                        #atom_list.append(atoms)
                        #print(atom_list)
            #for line in file:
                if 'phiA_0' in line:   
                    parts = line.split('=')   
                    if len(parts) > 1:
                        atoms_part = parts[1]
                        atoms = [atom.strip() for atom in atoms_part.split(',')]
                        atoms = [int(num) for num in atoms]
                        res.append(atoms[0])
                        #print(res)
                        #atom_list.append(atom.strip() for atom in atoms)
                if 'phiC_0' in line:

                    
                    parts = line.split('=')
                    if len(parts) > 1:
                        atoms_part = parts[1]
                        atoms = [atom.strip() for atom in atoms_part.split(',')]
                        atoms = atoms[1:]
                      
                        atoms = [int(num) for num in atoms]
                       
                        lig=atoms
                        #print(lig)
                        #atoms = match.group(1).split(',')
                        #atom_list.append(atom.strip() for atom in atoms)
        #with open(plumed_input_file, 'r') as file:         
          
 



        six_atoms=[]
        atom_list=lig+res
        
        six_atoms.append(atom_list)
        print(six_atoms)
        six_atoms = [list(i) for i in set(tuple(h) for h in six_atoms)]
      
        self.muti_six_atm_lst = six_atoms

if __name__ == '__main__':
    #amber
    test_ = Res_atom_select('protein.rst7', 'protein.prmtop', 'plumed.dat', 'Colvar', 100)
    #gromacs
    #test_ = Res_atom_select('complex.gro', None, 'plumed.dat', 'Colvar', 100)        
    test_.get_restr_plumed_input('MOL', True, ['lig_shape', 'HB_pair', 'HB_mainchain', 'Huggins'])

    ##after preliminary MD, do this
    # res_parm = test_.aly_traj_get_best_rest()
