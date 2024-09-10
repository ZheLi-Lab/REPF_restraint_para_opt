import time
import numpy as np
import math
import copy
from alive_progress import alive_bar

try:
    import openmm
    from openmm import unit
    from openmm import app
    from openmm import Platform
except ImportError:  # Openmm < 7.6
    from simtk import openmm
    from simtk import unit
    from simtk.openmm import app
    from simtk.openmm import Platform
from openmmtools import alchemy, integrators, states
from openmmplumed import PlumedForce
from sys import stdout
from .mdsystem import GromacsExplicitSystem,AmberExplicitSystem
from .alchemd import NormalMD,AlchemMD
from .Restraints_Select import Res_atom_select, RestraintParam
from mdtraj.reporters import NetCDFReporter
import json
import os 
import pandas as pd

def _report(info, if_savetraj=True, savetraj_freq=5000, reportstate_freq=1000):
    log_file=open(f'mdinfo_{info}', 'a')
    reporters=[]
    if if_savetraj:
        reporters.append(NetCDFReporter(f'output_{info}.nc', savetraj_freq))#traj output frequence
    reporters.append(app.StateDataReporter(log_file, reportstate_freq, step=True,potentialEnergy=True, temperature=True, volume=True, density=True, speed=True))
    return reporters

class RunAlchemdSimulation:
    def __init__(self,input_data, complex_coor, complex_topo):
        self.input_data=input_data

        self.complex_coor=complex_coor
        self.complex_topo=complex_topo
        # self.complex_coor = 'protein.rst7'
        # self.complex_topo = 'protein.prmtop'

        self.input_normal_MD = self.input_data.get_normalmd()
        self.input_restraint = self.input_data.get_restraint()
        self.input_alchemical = self.input_data.get_alchemical()

        self.if_normal_MD = self.input_normal_MD['normalmd']
        self.if_restraint = self.input_restraint['restraint']
        self.if_alchemical = self.input_alchemical['alchemical']

        if self.input_restraint['crd'] is None:
            self.input_restraint['crd'] = self.complex_coor
        if self.input_restraint['top'] is None:
            self.input_restraint['top'] = self.complex_topo
        self.input_alchemical['restraint_parm'] = None
        self.input_alchemical['another_restraint_parm'] = None

        self.state = None
        self.SystemContainer = AmberExplicitSystem(top=self.complex_topo, crd=self.complex_coor) 
        if self.if_restraint:
            self.RestraintRun=RestraintFlow(self.SystemContainer, self.input_restraint)
        if self.if_alchemical:
            if self.if_restraint:
                self.RestraintRun.run()
                self.state = self.RestraintRun.state
                self.input_alchemical['restraint_parm'] = self.RestraintRun.res_parm
                self.input_alchemical['another_restraint_parm'] = self.RestraintRun.another_res_parm
            print(self.input_alchemical)
            self.AlchemicalRun=AlchemicalFlow(self.SystemContainer, self.input_alchemical, )

    def run(self):

        if self.if_normal_MD:
            pass

        if self.if_restraint:
            self.RestraintRun.run()
            self.state = self.RestraintRun.state
            self.input_alchemical['restraint_parm'] = self.RestraintRun.res_parm
            self.input_alchemical['another_restraint_parm'] = self.RestraintRun.another_res_parm
            # self.RestraintRun.clean()

        if self.if_alchemical:
            self.AlchemicalRun.run()
            self.state = self.AlchemicalRun.state
            # self.AlchemicalRun.clean()
            



class RestraintFlow():
    def __init__(self, system_container, input_data):
        self.SystemContainer = system_container
        self.input_data = input_data
        self.state = None
        self.res_parm = None
        self.another_res_parm = None
        self.ligand_resname = self.input_data['ligand_resname']
        self.iflog_restraint_detail = self.input_data['iflog_restraint_detail']
        self.heat_nsteps = self.input_data['heat_nsteps']
        self.heat_iterations = self.input_data['heat_iterations']
        self.timestep = self.input_data['timestep']
        self.timestep_in_fs = self.input_data['timestep']*unit.femtoseconds
        self.f_plumed_input = self.input_data['f_plumed_input']
        self.density_nsteps = self.input_data['density_nsteps']
        self.npt_nsteps = self.input_data['npt_nsteps']
        self.f_npt_state = self.input_data['f_npt_state']
        self.crd = self.input_data['crd']
        self.top = self.input_data['top']
        self.f_plumed_output = self.input_data['f_plumed_output']
        self.plumed_freq = self.input_data['plumed_freq']
        self.lambdas_groups = json.load(open(self.input_data['lambdas_json']))[self.input_data['lambdas_group']]
        self.fake_state_xml = self.input_data['fake_state_xml']
        self.first_state_csv = self.input_data['first_state_csv']
        self.save_traj = self.input_data['save_traj']
        self.f_restraint = self.input_data['f_restraint'] # filename of the restraint parm file
        self.f_restraint2 = self.input_data['f_restraint2']
        self.res_sele_strategy = self.input_data['res_sele_strategy'].strip().split('|') #'lig_shape|HB_pair|HB_mainchain|Huggins'
        self.fix_lig_3atoms_index = [self.input_data['fix_lig_3atoms_index'].strip().split('|') if isinstance(self.input_data['fix_lig_3atoms_index'], str) else []][0] # '100|101|102'
        self.fix_lig_3atoms_index = [ int(i) for i in self.fix_lig_3atoms_index ]
        self.opt_cost_name = self.input_data['opt_cost_name']
        self.if_mean = self.input_data['if_mean']
        self.if_init_pose = self.input_data['if_init_pose']
        self.state = self.input_data['preliminary_md_inital_state']
        self.preliminary_min_and_heat = self.input_data['preliminary_min_and_heat']
        if self.opt_cost_name == False:
            self.opt_cost_name = None
        if self.save_traj == False:
            self.reportstate_freq = None
            self.savetraj_freq = None
        else:
            self.reportstate_freq = self.input_data['reportstate_freq']
            self.savetraj_freq = self.input_data['savetraj_freq']

        
    def run_nvt_min_and_heat_MD(self):
        nvt_md=NormalMD(self.SystemContainer,timestep=self.timestep_in_fs)
        if self.save_traj:
            nvt_md.reporters=_report('nvt', if_savetraj=True, savetraj_freq=self.savetraj_freq, reportstate_freq=self.reportstate_freq)
        nvt_md.minimize_cascade('com')
        nvt_md.heat_cascade(nsteps=self.heat_nsteps,heat_iterations=self.heat_iterations)
        self.state=nvt_md.final_state
        
        del nvt_md
        # return nvt_md.final_state
    
    def run_npt_md(self):
        npt_md_density=NormalMD(self.SystemContainer,timestep=self.timestep_in_fs,plumed=None,read_state=self.state,constant_pressure=1*unit.atmosphere)
        if self.save_traj:
            npt_md_density.reporters=_report('npt_density', if_savetraj=True, savetraj_freq=self.savetraj_freq, reportstate_freq=self.reportstate_freq)
        npt_md_density.density_run(self.density_nsteps)
        del npt_md_density

        npt_md=NormalMD(self.SystemContainer,timestep=self.timestep_in_fs,plumed=self.f_plumed_input,read_state=self.state,constant_pressure=1*unit.atmosphere)
        if self.save_traj:
            npt_md.reporters=_report('npt', if_savetraj=True, savetraj_freq=self.savetraj_freq, reportstate_freq=self.reportstate_freq)
        npt_md.run(self.npt_nsteps)
        npt_md.saveState(self.f_npt_state)
        self.state=npt_md.final_state
        del npt_md

    def gen_restraint_plumed_input(self):
        self.Restr_test = Res_atom_select(self.crd, self.top, self.f_plumed_input, self.f_plumed_output, self.plumed_freq)
        #gromacs restraint_selection
        #test_ = Res_atom_select('complex.gro', None, 'plumed.dat', 'Colvar', 100)
        self.Restr_test.get_restr_plumed_input(self.ligand_resname, self.iflog_restraint_detail, self.res_sele_strategy, self.fix_lig_3atoms_index)

    def clean(self):
        pass

    def run(self):
        #amber restraint_selection
        if os.path.exists(self.f_restraint):
            # Reading the existing res_parm
            res_df = pd.read_csv(self.f_restraint)
            self.res_parm = RestraintParam(eval(res_df.iloc[0, 2]), eval(res_df.iloc[0, 3]), float(res_df.iloc[0, 4]), float(res_df.iloc[0, 5]), float(res_df.iloc[0, 6]), float(res_df.iloc[0, 7]), float(res_df.iloc[0, 8]), float(res_df.iloc[0, 9]))
            if os.path.exists(self.f_restraint2):
                res_df2 = pd.read_csv(self.f_restraint2)
                self.another_res_parm = RestraintParam(eval(res_df2.iloc[0, 2]), eval(res_df2.iloc[0, 3]), float(res_df2.iloc[0, 4]), float(res_df2.iloc[0, 5]), float(res_df2.iloc[0, 6]), float(res_df2.iloc[0, 7]), float(res_df2.iloc[0, 8]), float(res_df2.iloc[0, 9]))
            self.state=self.f_npt_state
        else:
            if os.path.exists(self.f_npt_state) and os.path.exists(self.f_plumed_output):
                # Preliminary MD finish! will not run normal MD
                self.gen_restraint_plumed_input()
            else:
                if self.preliminary_min_and_heat:
                    self.run_nvt_min_and_heat_MD()
                self.gen_restraint_plumed_input()
                self.run_npt_md()
            self.res_parm = self.Restr_test.aly_traj_get_best_rest(self.lambdas_groups, self.fake_state_xml, self.first_state_csv, self.npt_nsteps, self.timestep, self.opt_cost_name, self.if_mean, self.if_init_pose)



class AlchemicalFlow(): #TODO: handle current_group_nb and current_group_chg
    def __init__(self, system_container, input_data, ):
        self.SystemContainer = system_container
        self.input_data = input_data
        self.lambdas_group=json.load(open(self.input_data['lambdas_json']))[self.input_data['lambdas_group']]
        self.state = self.input_data['input_state']
        self.pdbx = self.input_data['pdbx']# the file of the pdbx file

        self.set_rbfe_exception =self.input_data['set_rbfe_exception'] # added by ylzhong
        self.alchemical_co_ion = self.input_data['alchemical_co_ion'] # added by rdliu 
        if self.alchemical_co_ion:
            self.alchemical_co_ion = int(self.alchemical_co_ion)
        else:
            self.alchemical_co_ion = False

        if self.pdbx is None:
            self.current_group_nb = None
            self.current_group_chg = None
        else:
            self.current_group_nb = self.input_data['current_group_nb']
            self.current_group_chg = self.input_data['current_group_chg'] 
        self.restraint_parm = self.input_data['restraint_parm']# the Restraint_parm object generated by RestraintFlow
        self.another_restraint_parm = self.input_data['another_restraint_parm']
        if self.restraint_parm == None:
            print('Warning no restraint_parm is given!')
        else:
            self.restraint_parm.kbond = self.input_data['kbond']
            self.restraint_parm.ktheta1 = self.input_data['ktheta1']
            self.restraint_parm.ktheta2 = self.input_data['ktheta2']
            self.restraint_parm.kphi1 = self.input_data['kphi1']
            self.restraint_parm.kphi2 = self.input_data['kphi2']
            self.restraint_parm.kphi3 = self.input_data['kphi3']
            print('Complete the update of the restraint_parm object!')
        if self.another_restraint_parm is not None:
            self.another_restraint_parm.kbond = self.input_data['kbond']
            self.another_restraint_parm.ktheta1 = self.input_data['ktheta1']
            self.another_restraint_parm.ktheta2 = self.input_data['ktheta2']
            self.another_restraint_parm.kphi1 = self.input_data['kphi1']
            self.another_restraint_parm.kphi2 = self.input_data['kphi2']
            self.another_restraint_parm.kphi3 = self.input_data['kphi3']
            print('Complete the update of the another_restraint_parm object!')
        self.nsteps = self.input_data['nsteps']
        self.niterations = self.input_data['niterations']
        self.timestep_in_fs = self.input_data['timestep']*unit.femtoseconds
        self.save_traj = self.input_data['save_traj']
        self.if_min_heat_density = self.input_data['if_min_heat_density']
        self.simulation_lambdas_name = self.input_data['simulation_lambdas_name']
        self.annihilate_electrostatics = self.input_data['annihilate_electrostatics']
        self.cal_adj_ene_num = self.input_data['cal_adj_ene_num']
        alchem_group = 'resname MOL'
        if self.cal_adj_ene_num == 'all':
            self.cal_all_ene = True
        else:
            self.cal_all_ene = False
        if self.simulation_lambdas_name is None:
            self.simulation_lambdas = False
        else:
            self.simulation_lambdas = json.load(open(self.input_data['lambdas_json']))[self.simulation_lambdas_name]
        if self.save_traj == False:
            self.reportstate_freq = None
            self.savetraj_freq = None
        else:
            self.reportstate_freq = self.input_data['reportstate_freq']
            self.savetraj_freq = self.input_data['savetraj_freq']
        if self.alchemical_co_ion:
            alchem_group = f'resname MOL or resid {self.alchemical_co_ion}'

        if self.set_rbfe_exception:
            alchem_group = 'resname LAM LBM' # added by ylzhong

        print(self.restraint_parm)

        self.alchem_md=AlchemMD(self.SystemContainer,read_state=self.state,restraint_parm=self.restraint_parm, timestep=self.timestep_in_fs,
                        constant_pressure=1*unit.atmosphere, 
                        pdbx=self.pdbx, current_group_nb=self.current_group_nb , current_group_chg=self.current_group_chg,
                        annihilate_electrostatics = self.annihilate_electrostatics, set_rbfe_exception=self.set_rbfe_exception,
                                alchem_group=alchem_group, another_restraint_parm=self.another_restraint_parm)
        # added by ylzhong:set_rbfe_excption=self.set_rbfe_excption
        # is_rbfe


    def run_nvt_min_and_heat_MD(self):
        nvt_md=NormalMD(self.SystemContainer,timestep=self.timestep_in_fs)
        if self.save_traj:
            nvt_md.reporters=_report('nvt', if_savetraj=True, savetraj_freq=self.savetraj_freq, reportstate_freq=self.reportstate_freq)
        nvt_md.minimize_cascade('lig')
        nvt_md.heat_cascade(nsteps=2500,heat_iterations=10)
        self.state=nvt_md.final_state
        del nvt_md

    def run_npt_md(self):

        npt_md=NormalMD(self.SystemContainer,timestep=self.timestep_in_fs,plumed=None,read_state=self.state,constant_pressure=1*unit.atmosphere)
        if self.save_traj:
            npt_md.reporters=_report('npt', if_savetraj=True, savetraj_freq=self.savetraj_freq, reportstate_freq=self.reportstate_freq)
        npt_md.run(50000)
        npt_md.saveState('npt_final_state.xml')
        self.state=npt_md.final_state
        del npt_md

    def run(self, ):
        if self.if_min_heat_density:
            self.run_nvt_min_and_heat_MD()
            self.run_npt_md()
            self.alchem_md.set_state_xpv(self.state)

        self.alchem_md.set_lambdas(self.lambdas_group)

        if self.simulation_lambdas:
            self.alchem_md.set_simulation_lambdas(self.simulation_lambdas)
            
        if self.save_traj:
            self.alchem_md.reporters=_report('alc', if_savetraj=True, savetraj_freq=self.savetraj_freq, reportstate_freq=self.reportstate_freq)
        self.alchem_md.run(nsteps=self.nsteps, niterations=self.niterations, cal_all_ene=self.cal_all_ene, cal_adj_ene_num=self.cal_adj_ene_num)
        self.state=self.alchem_md.final_state
        self.alchem_md.saveState('alc_final_state.xml')


    
    def clean(self):
        del self.alchem_md


