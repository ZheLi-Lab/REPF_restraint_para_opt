try:
    import openmm
    from openmm import unit
    from openmm import app
except ImportError:  # Openmm < 7.6
    from simtk import openmm
    from simtk import unit
    from simtk.openmm import app

import copy,sys
import sys,os
import numpy as np
import mdtraj
from openmmtools import integrators, states, cache
from openmmtools.forcefactories import restrain_atoms_by_dsl
#from openmmtools import alchemy
from . import alchemy_by_group as alchemy
from .Restraints_Select.restraints import Boresch, RestraintParameterError, RestraintState
from .Restraints_Select.restraints2 import Boresch2, RestraintState2 
from .tools.tools import Timer
import pandas as pd
from .pdbx.pdbx_parser import PdbxParser

def get_adjacent_numbers(n, threshold, adjacent_num=5):
    result = []
    for i in range(max(0, n - adjacent_num), min(threshold, n + adjacent_num)+1):
        result.append(i)
    return result

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

class RestraintParam():
    def __init__(self, rec_atoms, lig_atoms, r, theta1, theta2, phi1, phi2, phi3, kbond=10, ktheta1=10, ktheta2=10, kphi1=10, kphi2=10, kphi3=10):
        self.rec_atoms = rec_atoms # [rec1, rec2, rec3]
        self.lig_atoms = lig_atoms # [lig1, lig2, lig3]
        self.r = r # distance between rec3 and lig1, unit in angstrom
        self.theta1 = theta1 # angle between rec2, rec3, lig1, unit in radians 
        self.theta2 = theta2 # angle between rec3, lig1, lig2  
        self.phi1 = phi1 # dihedral between rec1, rec2, rec3, lig1
        self.phi2 = phi2 # dihedral between rec2, rec3, lig1, lig2 
        self.phi3 = phi3 # dihedral between rec3, lig1, lig2, lig3 
        self.kbond = kbond # unit in kcal/mol/A^2
        # self.kangle = kangle # unit in kcal/mol/rad^2
        # self.kdihedral = kdihedral
        self.ktheta1 = ktheta1 # unit in kcal/mol/rad^2
        self.ktheta2 = ktheta2 # unit in kcal/mol/rad^2 for single atom restraint, this term to be zero
        self.kphi1 = kphi1 # unit in kcal/mol/rad^2
        self.kphi2 = kphi2 # unit in kcal/mol/rad^2 for single atom restraint, this term to be zero
        self.kphi3 = kphi3 # unit in kcal/mol/rad^2 for single atom restraint, this term to be zero
    def __repr__(self):
        return f'rec_atoms:{self.rec_atoms}, lig_atoms:{self.lig_atoms}, r:{self.r} A, theta1:{self.theta1} rad, theta2:{self.theta2} rad, phi1:{self.phi1} rad, phi2:{self.phi2} rad, phi3:{self.phi3} rad, kbond:{self.kbond} kcal/mol/A^2, K_theta1:{self.ktheta1} kcal/mol/rad^2, self.K_theta2:{self.ktheta2} kcal/mol/rad^2, self.K_phi1:{self.kphi1} kcal/mol/rad^2, self.K_phi2:{self.kphi2} kcal/mol/rad^2, self.K_phi3:{self.kphi3} kcal/mol/rad^2'

class MDClass(app.Simulation):
    def __init__(self, system_container):
        self.topology = system_container.topology 
        self.system = system_container.system 
        self.reporters = []
        self._usesPBC = True

    def set_state_xpv(self,state):
        #set context positions, velocities, box vectors based on state
        positions = state.getPositions()
        velocities = state.getVelocities()
        boxvectors = state.getPeriodicBoxVectors()
        self.context.setPositions(positions)
        self.context.setVelocities(velocities)
        self.context.setPeriodicBoxVectors(*boxvectors)

    def useplumed(self,plumed_filename):
        print("Using plumed file %s ..." % (plumed_filename))
        from openmmplumed import PlumedForce
        plumed_infile = open(plumed_filename,'r')
        try:
            plumed_script = plumed_infile.read()
        finally:
            plumed_infile.close()
        self.plumedforce = PlumedForce(plumed_script)
        self.system_container.system.addForce(self.plumedforce)

    @property
    def final_state(self):
        return self.context.getState(getPositions=True,getVelocities=True,enforcePeriodicBox=True)

    @property
    def sampler_state(self):
        return states.SamplerState.from_context(self.context)


class NormalMD(MDClass):
    def __init__(self, system_container, 
                timestep=4*unit.femtoseconds, temperature=298*unit.kelvin, 
                constant_pressure=None, plumed=None, read_state=None,
                platform=None, platformProperties=None):
        self.system_container=copy.deepcopy(system_container)
        self.original_system_container=system_container
        self.read_state=read_state
        self.timestep=timestep
        self.temperature=temperature
        self.constant_pressure=constant_pressure
        super().__init__(self.system_container)
        self.topology=self.original_system_container.topology #could not use self.system_container. Could be a bug
        if plumed:
            self.plumedforce = None
            self.useplumed(plumed)
        if constant_pressure:
            self.thermodynamic_state = states.ThermodynamicState(self.system_container.system, temperature, constant_pressure)
        else:
            self.thermodynamic_state = states.ThermodynamicState(self.system_container.system, temperature)
        self.integrator= integrators.LangevinIntegrator(timestep=self.timestep,splitting="V R R R O R R R V")
        self.integrator = openmm.openmm.LangevinMiddleIntegrator(self.temperature, 1/unit.picoseconds, self.timestep)#TO VERTIFY
        self.context=self.thermodynamic_state.create_context(self.integrator,platform=platform,platform_properties=platformProperties)
        # adding positional restraints will change thermodynamic_state. Save a copy
        self.thermodynamic_state_original=copy.deepcopy(self.thermodynamic_state)

        if not read_state:
            self.context.setPositions(self.system_container.positions)
        else:
            if isinstance(read_state, str):
                self.loadState(read_state) # change this to sampler_state
            else:
                self.set_state_xpv(read_state)



    def minimize(self):
        openmm.LocalEnergyMinimizer.minimize(self.context)

    def minimize_cascade(self, mode='com'):
        #minimize cascade, 
        # mode: 'com', perform minimization for a protein-ligand complex system, default
        # mode: 'lig', perform minimization for a ligand system
        if mode == 'com':
            self.minimize_cascade_com()
        elif mode == 'lig':
            self.minimize_cascade_lig()

    def minimize_cascade_lig(self):
        print('Starting 4 steps minimization cascade')
        print('Performing 1st minimization step...')
        #adding positional restraints will change thermodynamic_state
        restrain_atoms_by_dsl(self.thermodynamic_state, self.sampler_state, self.topology, '(all and not water) and not (name =~ "H.*")',sigma=0.1*unit.angstrom)
        #self.thermodynamic_state.K=100.0*unit.kilocalories_per_mole/unit.angstrom**2
        self.context.reinitialize(preserveState=True)
        self.minimize()

        print('Performing 2rd minimization step...')
        #remove positional restraint forces.
        self.thermodynamic_state=copy.deepcopy(self.thermodynamic_state_original)
        self.context.reinitialize(preserveState=True)
        self.minimize()

    def minimize_cascade_com(self):
        print('Starting 4 steps minimization cascade')
        print('Performing 1st minimization step...')
        #adding positional restraints will change thermodynamic_state
        restrain_atoms_by_dsl(self.thermodynamic_state, self.sampler_state, self.topology, '(all and not water) and not (name =~ "H.*")',sigma=0.1*unit.angstrom)
        #self.thermodynamic_state.K=100.0*unit.kilocalories_per_mole/unit.angstrom**2
        self.context.reinitialize(preserveState=True)
        self.minimize()

        print('Performing 2nd minimization step...')
        #add a new type of positional restraints, first we need to restore the original thermodynamic_state
        self.thermodynamic_state=copy.deepcopy(self.thermodynamic_state_original)
        restrain_atoms_by_dsl(self.thermodynamic_state, self.sampler_state, self.topology, '(all and not water and not resname MOL) and not (name =~ "H.*")',sigma=0.1*unit.angstrom)
        #self.thermodynamic_state.K=100.0*unit.kilocalories_per_mole/unit.angstrom**2
        self.context.reinitialize(preserveState=True)
        self.minimize()

        print('Performing 3rd minimization step...')
        #add a new type of positional restraints, first we need to restore the original thermodynamic_state
        self.thermodynamic_state=copy.deepcopy(self.thermodynamic_state_original)
        restrain_atoms_by_dsl(self.thermodynamic_state, self.sampler_state, self.topology, 'backbone and not (name =~ "H.*")',sigma=0.1*unit.angstrom)
        #self.thermodynamic_state.K=100.0*unit.kilocalories_per_mole/unit.angstrom**2
        self.context.reinitialize(preserveState=True)
        self.minimize()

        print('Performing 4th minimization step...')
        #remove positional restraint forces.
        self.thermodynamic_state=copy.deepcopy(self.thermodynamic_state_original)
        self.context.reinitialize(preserveState=True)
        self.minimize()

    def heat_cascade(self,nsteps=50000,heat_iterations=60):
        print('Starting heat cascade, total steps %d, heat iterations %d' % (nsteps, heat_iterations))
        self.thermodynamic_state=copy.deepcopy(self.thermodynamic_state_original)
        restrain_atoms_by_dsl(self.thermodynamic_state, self.sampler_state, self.topology, 'all and not water')
        self.thermodynamic_state.K=10.0*unit.kilocalories_per_mole/unit.angstrom**2
        self.context.reinitialize(preserveState=True)
        T_increment=self.temperature/heat_iterations
        for i in range(heat_iterations):
            T=(i+1)*T_increment
            self.integrator.setTemperature(T*unit.kelvin)
            self.run(int(nsteps/heat_iterations), report_speed=False)

    def density_run(self, nsteps=10000, mode='com'):
        print('Starting density equilibration, total steps %d' % (nsteps))
        #add positional restraints
        if not self.constant_pressure:
            raise ValueError('Density run must be performed under constant pressure')
        self.thermodynamic_state=copy.deepcopy(self.thermodynamic_state_original)

        # Bug will occur when 'protein' represents more than molecules.
        # TODO: change to positional restraint may solve this problem.
        if mode == 'com':
            try:
                restrain_atoms_by_dsl(self.thermodynamic_state, self.sampler_state, self.topology, 'protein')
            except:
                restrain_atoms_by_dsl(self.thermodynamic_state, self.sampler_state, self.topology, 'resname MOL')
        elif mode == 'lig':
            restrain_atoms_by_dsl(self.thermodynamic_state, self.sampler_state, self.topology, 'resname MOL')

        self.thermodynamic_state.K=10.0*unit.kilocalories_per_mole/unit.angstrom**2
        self.context.reinitialize(preserveState=True)
        self.run(nsteps)
        #remove positional restraints
        self.thermodynamic_state.K=0.0*unit.kilocalories_per_mole/unit.angstrom**2
        self.thermodynamic_state.apply_to_context(self.context)

    def run(self,nsteps,report_speed=True):
        if report_speed:
            t=Timer()
            t.start()

        self._simulate(endStep=self.currentStep+nsteps)
        
        if report_speed:
            clock_time=t.stop()
            ns_per_day=self.timestep.value_in_unit(unit=unit.nanosecond)*nsteps*86400/clock_time
            print("Simulation Speed: %4.3f ns/day" % (ns_per_day))


class AlchemMD(MDClass):
    # pdbx is used to specify group vdw decouple and group charge annihilation
    def __init__(self, system_container, restraint_parm=None,
                timestep=4*unit.femtoseconds, temperature=298*unit.kelvin, 
                pdbx=None,
                current_group_nb=0, current_group_chg=0,
                constant_pressure=None, plumed=None, read_state=None, 
                annihilate_electrostatics=True,
                alchem_group='resname MOL',platform=None, platformProperties=None,
                set_rbfe_exception=False, another_restraint_parm=None):
        self.alchemical_state=None
        self.restraint_state=None
        self.timestep=timestep
        self.temperature = temperature
        self.context_cache=cache.ContextCache(platform=platform,platform_properties=platformProperties)
        self.original_system_container=system_container
        self.system_container=copy.deepcopy(system_container)
        self.restraint_parm=restraint_parm
        self.another_restraint_parm=another_restraint_parm
        self.read_state=read_state
        self.u_kln_array = None
        self.u_kln = None
        self.lambdas={}
        # added by rdliu
        self.simulation_lambdas={}
        self.simulation_lambdas_idxs=[]

        self.set_rbfe_exception=set_rbfe_exception
        #added by ylzhong

        self.nsteps=0
        super().__init__(self.system_container)
        self.topology=self.original_system_container.topology

        self.current_group_nb=current_group_nb 

        self.pdbx=pdbx
        if pdbx is not None:
            alchem_start_id=self.get_alchem_start_id(self.original_system_container.topology, alchem_group)
            self.pdbx=PdbxParser(pdbx)
            print("alchem_start_id: %s" % (alchem_start_id))
            decouple_groups_nb=self.pdbx.get_group_nb_dict(alchem_start_id)
            initial_charge, target_charge = self.pdbx.get_charge_list(alchem_start_id,col=current_group_chg)
            print(decouple_groups_nb)
            print(initial_charge, target_charge)
        else:
            decouple_groups_nb=False
            initial_charge=False
            target_charge=False

#self.timestep.value_in_unit(unit=unit.femtoseconds)

        if plumed:
            self.useplumed(plumed)
        if constant_pressure:
            self.thermodynamic_state = states.ThermodynamicState(self.system_container.system, temperature, constant_pressure)
        else:
            self.thermodynamic_state = states.ThermodynamicState(self.system_container.system, temperature)

        self.alchemical_state = self.create_alchem_state(alchem_group, annihilate_electrostatics=annihilate_electrostatics, 
                                                        set_initial_charge=initial_charge, set_target_charge=target_charge,
                                                        decouple_groups_nb=decouple_groups_nb, current_group_nb=current_group_nb,
                                                        set_rbfe_exception=set_rbfe_exception) #init self.alchemical_state
        # modded by ylzhong set_rbfe_exception
        # determine whether we need to add restraint
        if self.restraint_parm is not None:
            self.restraint_state = self.create_restrain_state(restraint_parm) #init self.restraint_state
            composable_states = [self.alchemical_state, self.restraint_state]
            if self.another_restraint_parm is not None:
                self.another_restraint_state = self.create_restrain_state(self.another_restraint_parm,True)
                composable_states.append(self.another_restraint_state)            
        else:
            composable_states = [self.alchemical_state]

        self.compound_state = states.CompoundThermodynamicState(
                         thermodynamic_state=self.thermodynamic_state, composable_states=composable_states)

        self.integrator= integrators.LangevinIntegrator(timestep=self.timestep,splitting="V R R R O R R R V")
        self.integrator = openmm.openmm.LangevinMiddleIntegrator(self.temperature, 1/unit.picoseconds, self.timestep)
        self.context, self.integrator = self.context_cache.get_context(self.compound_state, self.integrator)

        if not read_state:
            self.context.setPositions(self.system_container.positions)
        else:
            if isinstance(read_state, str):
                self.loadState(read_state) # change this to sampler_state
            else:
                self.set_state_xpv(read_state)
    
    def get_alchem_start_id(self, topology, alchem_group):
        if isinstance(topology, mdtraj.Topology):
            mdtraj_topology = topology
        else:
            mdtraj_topology = mdtraj.Topology.from_openmm(topology)
        # Determine indices of the atoms to restrain.
        alchem_atoms = mdtraj_topology.select(alchem_group).tolist()
        return alchem_atoms[0]

    def set_lambdas(self, lambdas):
        self.lambdas=lambdas

        # assert the length of all lambdas to be the same
        len_prev_lambda = -1 # we do not have the length of previous lambda here
        for lambda_type in self.lambdas:
            len_current_lambda = len(self.lambdas[lambda_type])
            if len_prev_lambda > -1:
                if len_prev_lambda != len_current_lambda:
                    raise ValueError('length of lambdas should be the same')
            len_prev_lambda = len_current_lambda

    # added by rdliu
    def set_simulation_lambdas(self, lambdas):
        self.simulation_lambdas = lambdas
        # assert the length of all lambdas to be the same
        len_prev_lambda = -1 # we do not have the length of previous lambda here
        for lambda_type in self.simulation_lambdas:
            len_current_lambda = len(self.simulation_lambdas[lambda_type])
            if len_prev_lambda > -1:
                if len_prev_lambda != len_current_lambda:
                    raise ValueError('length of lambdas should be the same')
            len_prev_lambda = len_current_lambda
        # get the keyname startwith "lambda_"
        compute_mbar_lambda_keys = [key for key in self.lambdas if key.startswith("lambda_")]
        simulation_lambda_keys = [key for key in self.simulation_lambdas if key.startswith("lambda_")]
        if compute_mbar_lambda_keys != simulation_lambda_keys:
            raise ValueError("Error! Your simulation_lambda_keys are not as same as the lambda_keys in the compute_mbar_lambda!")

        # judge if self.simulation_lambdas is the subset of self.lambdas
        compute_mbar_tuples = []
        simulation_tuples = []
        for i in range(len(self.lambdas[compute_mbar_lambda_keys[0]])):
            compute_tuple = tuple(self.lambdas[key][i] for key in compute_mbar_lambda_keys)
            compute_mbar_tuples.append(compute_tuple)
        for i in range(len(self.simulation_lambdas[compute_mbar_lambda_keys[0]])):
            simulation_tuple = tuple(self.simulation_lambdas[key][i] for key in compute_mbar_lambda_keys)
            simulation_tuples.append(simulation_tuple)   
        for item in simulation_tuples:
            # print(item, compute_mbar_tuples)
            if item not in compute_mbar_tuples:
                # print(item, compute_mbar_tuples)
                raise ValueError("Error! Your simulation lambda is not found in the mbar_lambda!")
            
        self.simulation_lambdas_idxs = [compute_mbar_tuples.index(item) for item in simulation_tuples]

    def change_state(self, state, nstate):
        for lambda_type in self.lambdas:
            setattr(state, lambda_type, self.lambdas[lambda_type][nstate])

    def get_state_lambda(self, nstate):
        lambda_types=[]
        lambda_values=[]
        for lambda_type in self.lambdas:
            lambda_value=self.lambdas[lambda_type][nstate]
            if lambda_type == 'lambda_electrostatics':
                lambda_value = np.around(1-lambda_value, decimals=3)
            elif lambda_type == 'lambda_sterics':
                lambda_value = np.around(1-lambda_value, decimals=3)
            lambda_types.append(lambda_type)
            lambda_values.append(lambda_value)
        return lambda_types, lambda_values

    def create_alchem_state(self, alchem_group, annihilate_electrostatics=True, set_initial_charge=False, set_target_charge=False, decouple_groups_nb=False, current_group_nb=0,
                            set_rbfe_exception=False):
        alchem_atoms = self.original_system_container.mdtraj_topology.select(alchem_group) # cannot use self.system_container, otherwise will raise an exception, which could be a bug
        alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=alchem_atoms,annihilate_electrostatics=annihilate_electrostatics)
        #print(alchemical_region.name)
        factory = alchemy.AbsoluteAlchemicalFactory()
        alchemical_system = factory.create_alchemical_system(self.thermodynamic_state.system, alchemical_region, 
                            set_initial_charge=set_initial_charge, set_target_charge=set_target_charge,
                            decouple_groups_nb=decouple_groups_nb, current_group_nb=current_group_nb,
                            set_rbfe_exception=set_rbfe_exception)
        # modded by ylzhong 2024.5.24

        self.thermodynamic_state.system = alchemical_system
        alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
        return alchemical_state

    def create_restrain_state(self, restraint, ifanother=False):
        FORCE_CONSTANT_BOND=restraint.kbond*unit.kilocalories_per_mole/unit.angstrom**2
        K_theta1_FORCE_CONSTANT_ANGLE=restraint.ktheta1*unit.kilocalories_per_mole/unit.radians**2
        K_theta2_FORCE_CONSTANT_ANGLE=restraint.ktheta2*unit.kilocalories_per_mole/unit.radians**2
        K_phi1_FORCE_CONSTANT_DIHE=restraint.kphi1*unit.kilocalories_per_mole/unit.radians**2
        K_phi2_FORCE_CONSTANT_DIHE=restraint.kphi2*unit.kilocalories_per_mole/unit.radians**2
        K_phi3_FORCE_CONSTANT_DIHE=restraint.kphi3*unit.kilocalories_per_mole/unit.radians**2
        if not ifanother:
            restraint = Boresch(restrained_receptor_atoms=restraint.rec_atoms,
                                    restrained_ligand_atoms=restraint.lig_atoms,
                                    K_r=FORCE_CONSTANT_BOND, r_aA0=restraint.r*unit.angstrom,
                                    K_thetaA=K_theta1_FORCE_CONSTANT_ANGLE, K_thetaB=K_theta2_FORCE_CONSTANT_ANGLE,
                                    theta_A0=restraint.theta1*unit.radians, theta_B0=restraint.theta2*unit.radians,
                                    K_phiA=K_phi1_FORCE_CONSTANT_DIHE, K_phiB=K_phi2_FORCE_CONSTANT_DIHE, K_phiC=K_phi3_FORCE_CONSTANT_DIHE,
                                    phi_A0=restraint.phi1*unit.radians, phi_B0=restraint.phi2*unit.radians, phi_C0=restraint.phi3*unit.radians
            )
            restraint.restrain_state(self.thermodynamic_state)
            restraint_state = RestraintState(lambda_restraints=0.0)
        else:
            restraint = Boresch2(restrained_receptor_atoms=restraint.rec_atoms,
                                    restrained_ligand_atoms=restraint.lig_atoms,
                                    K_r=FORCE_CONSTANT_BOND, r_aA0=restraint.r*unit.angstrom,
                                    K_thetaA=K_theta1_FORCE_CONSTANT_ANGLE, K_thetaB=K_theta2_FORCE_CONSTANT_ANGLE,
                                    theta_A0=restraint.theta1*unit.radians, theta_B0=restraint.theta2*unit.radians,
                                    K_phiA=K_phi1_FORCE_CONSTANT_DIHE, K_phiB=K_phi2_FORCE_CONSTANT_DIHE, K_phiC=K_phi3_FORCE_CONSTANT_DIHE,
                                    phi_A0=restraint.phi1*unit.radians, phi_B0=restraint.phi2*unit.radians, phi_C0=restraint.phi3*unit.radians
            )
            restraint.restrain_state(self.thermodynamic_state)
            restraint_state = RestraintState2(lambda_restraints2=0.0)
        return restraint_state

    def set_lambda_contexts(self):
    # This function is useless
        try:
            lambda_sterics=self.lambda_sterics
            lambda_electrostatics=self.lambda_electrostatics
            lambda_restraints=self.lambda_restraints
        except:
            print("lambdas not set, please call set_lambdas method first")
            sys.exit(1)

        self.lambda_contexts=[]
        nstates=len(lambda_sterics)
        for k in range(nstates):
            context, integrator = self.context_cache.get_context(self.compound_state, self.integrator)
            self.compound_state.lambda_electrostatics = lambda_electrostatics[k] # 0 means completely decoupled
            self.compound_state.lambda_sterics = lambda_sterics[k]
            self.compound_state.lambda_restraints = lambda_restraints[k] 
            self.compound_state.apply_to_context(context)
            self.lambda_contexts.append(context)
        

    def run(self, nsteps, niterations, save_state=True, current_state=None, cal_all_ene=False, cal_adj_ene_num=5):
        # 
        # nsteps: run simulation for nsteps and then calculate deltaU, this is called one iteration
        # niteration: how many iterations will be calculated in total for each state
        # current_state: calculate a specific state, if not specified, all states will be calculated
        # cal_all_ene: If set to be True, then calculate the deltaU with respect to all states. 
        #              Otherwise only deltaU of adjacent states will be calculated.

        self.nsteps=nsteps
        kT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * self.integrator.getTemperature()

        nstates=len(list(self.lambdas.values())[0])
        all_states = range(nstates)
        # modified by rdliu
        if not current_state:
            if len(self.simulation_lambdas_idxs) == 0:
                states = all_states
            else:
                states = self.simulation_lambdas_idxs       
        else: states = current_state

        state_skipped=False

        timestep = self.timestep.value_in_unit(unit=unit.femtoseconds)
        for k in states:            
            # progress bar
            try:
                from tqdm import tqdm
                # iterator = tqdm(range(niterations), ascii=True, desc="Alchemical state %3d" % (k))
                iterator = tqdm(range(niterations), ascii=True, desc="Alchemical state %3d" % (k),
                                unit='fs', unit_scale=self.nsteps*timestep)
            except:
                iterator = progressbar(range(niterations),           "Alchemical state %3d" % (k), 40)
            
            if save_state:
                if self.pdbx is not None:
                    prefix='state_g'+str(self.current_group_nb)+'_'
                else: prefix='state_'
                statefile=prefix+'s'+str(k)+'.xml'

            # skip state that has already finished
            if save_state and os.path.exists(statefile):
                last_state_file=statefile
                state_skipped=True
                print('State file %s already exists, skip this state', statefile)
                continue
            if state_skipped:
                self.loadState(last_state_file) 

            # calculate energy with respect to all lambdas or just adjacent lambdas
            if cal_all_ene:
                ene_states=all_states
            else:
                ene_states=get_adjacent_numbers(k, nstates-1, cal_adj_ene_num)
            n_ene_states = len(ene_states)
            print(ene_states)

            #Initialize dU pandas dataframe
            simulation_lambda_types, simulation_lambda_values = self.get_state_lambda(k)
            simulation_lambda_types.insert(0, 'times(ps)')
            # print(muti_idx_names)
            times_lambda_tuples = [(i*nsteps*timestep/1000,)+tuple(simulation_lambda_values)  for i in range(0, niterations)]
            muti_idx = pd.MultiIndex.from_tuples(times_lambda_tuples, names=simulation_lambda_types)
            zero_shape = np.zeros((niterations, n_ene_states))
            # columns_ = [tuple(self.get_state_lambda(l)[1]) for l in range(0,n_ene_states)]
            columns_ = [tuple(self.get_state_lambda(l)[1]) for l in ene_states] # fix bug for IndexError: iloc cannot enlarge its target object
            single_simulation_df = pd.DataFrame(zero_shape, columns=columns_)
            single_simulation_df.index = muti_idx
            # print(single_simulation_df.columns)
            # print(single_simulation_df.shape)
            # added by phologlucinol, to align the ene_states to the index of single_simulation_df
            index_gap = max(ene_states)+1-n_ene_states
            # start simulation for this state
            for iteration in iterator:
                self.change_state(self.compound_state, k)
                self.compound_state.apply_to_context(self.context)
                self._simulate(endStep=self.currentStep+nsteps)
                # calcualte dU
                for l in ene_states:
                    if l > all_states[-1] : continue
                    if l < all_states[0] : continue
                    self.change_state(self.compound_state, l)
                    self.compound_state.apply_to_context(self.context)
                    single_simulation_df.iloc[iteration, l-index_gap] = self.context.getState(getEnergy=True).getPotentialEnergy() / kT

            # save state and dU files
            if save_state: self.saveState(statefile)

            if self.pdbx is not None:
                csv_prefix='state_g'+str(self.current_group_nb)+'_'
            else: csv_prefix='state_'
            csv_file=csv_prefix+'s'+str(k)+'.csv'
            single_simulation_df.to_csv(csv_file, sep="|")
            # np.save(dU_file, self.u_kln_array)

        #self.u_kln = self.format_u_kln(self.u_kln_array)

    @property
    def final_state(self):
        return self.context.getState(getPositions=True,getVelocities=True,enforcePeriodicBox=True)


