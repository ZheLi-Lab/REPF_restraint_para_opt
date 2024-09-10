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
from openmmtools import integrators, states, cache
from openmmtools.forcefactories import restrain_atoms_by_dsl
#from openmmtools import alchemy
from . import alchemy_by_group as alchemy
from .restraints import Boresch, RestraintParameterError, RestraintState
from .tools import Timer
import pandas as pd

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
    def __init__(self, rec_atoms, lig_atoms, r, theta1, theta2, phi1, phi2, phi3, kbond=10, kangle=10, kdihedral=10):
        self.rec_atoms = rec_atoms # [rec1, rec2, rec3]
        self.lig_atoms = lig_atoms # [lig1, lig2, lig3]
        self.r = r # distance between rec3 and lig1, unit in angstrom
        self.theta1 = theta1 # angle between rec2, rec3, lig1, unit in radians
        self.theta2 = theta2 # angle between rec3, lig1, lig2
        self.phi1 = phi1 # dihedral between rec1, rec2, rec3, lig1
        self.phi2 = phi2 # dihedral between rec2, rec3, lig1, lig2
        self.phi3 = phi3 # dihedral between rec3, lig1, lig2, lig3
        self.kbond = kbond # unit in kcal/mol/A^2
        self.kangle = kangle # unit in kcal/mol/rad^2
        self.kdihedral = kdihedral

class MDClass(app.Simulation):
    def __init__(self, system_container):
        self.topology = system_container.topology 
        self.system = system_container.system 
        self.reporters = []
        self._usesPBC = True

    def set_state_xpv(self,state,context):
        #set context positions, velocities, box vectors based on state
        positions = state.getPositions()
        velocities = state.getVelocities()
        boxvectors = state.getPeriodicBoxVectors()
        context.setPositions(positions)
        context.setVelocities(velocities)
        context.setPeriodicBoxVectors(*boxvectors)

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
        self.context=self.thermodynamic_state.create_context(self.integrator,platform=platform,platform_properties=platformProperties)
        # adding positional restraints will change thermodynamic_state. Save a copy
        self.thermodynamic_state_original=copy.deepcopy(self.thermodynamic_state)

        if not read_state:
            self.context.setPositions(self.system_container.positions)
        else:
            if isinstance(read_state, str):
                self.loadState(read_state) # change this to sampler_state
            else:
                self.set_state_xpv(read_state,self.context)



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
            restrain_atoms_by_dsl(self.thermodynamic_state, self.sampler_state, self.topology, 'protein')
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
    def __init__(self, system_container, restraint_parm=None,
                timestep=4*unit.femtoseconds, temperature=298*unit.kelvin, 
                decouple_groups_nb=False, current_group_nb=0,
                constant_pressure=None, plumed=None, read_state=None, 
                annihilate_electrostatics=True,
                alchem_group='resname MOL',platform=None, platformProperties=None):
        self.alchemical_state=None
        self.restraint_state=None
        self.timestep=timestep
        self.context_cache=cache.ContextCache(platform=platform,platform_properties=platformProperties)
        self.original_system_container=system_container
        self.system_container=copy.deepcopy(system_container)
        self.restraint_parm=restraint_parm
        self.read_state=read_state
        self.decouple_groups_nb=decouple_groups_nb
        self.current_group_nb=current_group_nb
        self.u_kln_array = None
        self.u_kln = None
        self.lambdas={}
        self.nsteps=0
        super().__init__(self.system_container)
        self.topology=self.original_system_container.topology

#self.timestep.value_in_unit(unit=unit.femtoseconds)

        if plumed:
            self.useplumed(plumed)
        if constant_pressure:
            self.thermodynamic_state = states.ThermodynamicState(self.system_container.system, temperature, constant_pressure)
        else:
            self.thermodynamic_state = states.ThermodynamicState(self.system_container.system, temperature)

        self.alchemical_state = self.create_alchem_state(alchem_group, annihilate_electrostatics=annihilate_electrostatics, decouple_groups_nb=decouple_groups_nb, current_group_nb=current_group_nb) #init self.alchemical_state

        # determine whether we need to add restraint
        if self.restraint_parm is not None:
            self.restraint_state = self.create_restrain_state(restraint_parm) #init self.restraint_state
            composable_states = [self.alchemical_state, self.restraint_state]
        else:
            composable_states = [self.alchemical_state]

        self.compound_state = states.CompoundThermodynamicState(
                         thermodynamic_state=self.thermodynamic_state, composable_states=composable_states)

        self.integrator= integrators.LangevinIntegrator(timestep=self.timestep,splitting="V R R R O R R R V")
        self.context, self.integrator = self.context_cache.get_context(self.compound_state, self.integrator)

        if not read_state:
            self.context.setPositions(self.system_container.positions)
        else:
            if isinstance(read_state, str):
                self.loadState(read_state) # change this to sampler_state
            else:
                self.set_state_xpv(read_state,self.context)

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

    def change_state(self, state, nstate):
        for lambda_type in self.lambdas:
            setattr(state, lambda_type, self.lambdas[lambda_type][nstate])

    def get_state_lambda(self, nstate):
        lambda_types=[]
        lambda_values=[]
        for lambda_type in self.lambdas:
            lambda_value=self.lambdas[lambda_type][nstate]
            if lambda_type == 'lambda_electrostatics':
                lambda_value = np.around(1-lambda_value, decimals=2)
            elif lambda_type == 'lambda_sterics':
                lambda_value = np.around(1-lambda_value, decimals=2)
            lambda_types.append(lambda_type)
            lambda_values.append(lambda_value)
        return lambda_types, lambda_values

    def create_alchem_state(self, alchem_group, annihilate_electrostatics=True, decouple_groups_nb=False, current_group_nb=0):
        alchem_atoms = self.original_system_container.mdtraj_topology.select(alchem_group) # cannot use self.system_container, otherwise will raise an exception, which could be a bug
        alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=alchem_atoms,annihilate_electrostatics=annihilate_electrostatics)
        #print(alchemical_region.name)
        factory = alchemy.AbsoluteAlchemicalFactory()
        alchemical_system = factory.create_alchemical_system(self.thermodynamic_state.system, alchemical_region, 
                            decouple_groups_nb=decouple_groups_nb, current_group_nb=current_group_nb)

        self.thermodynamic_state.system = alchemical_system
        alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
        return alchemical_state

    def create_restrain_state(self, restraint):
        FORCE_CONSTANT_BOND=restraint.kbond*unit.kilocalories_per_mole/unit.angstrom**2
        FORCE_CONSTANT_ANGLE=restraint.kangle*unit.kilocalories_per_mole/unit.radians**2
        FORCE_CONSTANT_DIHE=restraint.kdihedral*unit.kilocalories_per_mole/unit.radians**2
        restraint = Boresch(restrained_receptor_atoms=restraint.rec_atoms,
                                restrained_ligand_atoms=restraint.lig_atoms,
                                K_r=FORCE_CONSTANT_BOND, r_aA0=restraint.r*unit.angstrom,
                                K_thetaA=FORCE_CONSTANT_ANGLE, K_thetaB=FORCE_CONSTANT_ANGLE,
                                theta_A0=restraint.theta1*unit.radians, theta_B0=restraint.theta2*unit.radians,
                                K_phiA=FORCE_CONSTANT_DIHE, K_phiB=FORCE_CONSTANT_DIHE, K_phiC=FORCE_CONSTANT_DIHE,
                                phi_A0=restraint.phi1*unit.radians, phi_B0=restraint.phi2*unit.radians, phi_C0=restraint.phi3*unit.radians
        )
        restraint.restrain_state(self.thermodynamic_state)
        restraint_state = RestraintState(lambda_restraints=0.0)
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
        

    def run(self, nsteps, niterations, save_state=True, current_state=None, cal_all_ene=False, ):
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
        if not current_state:
            states = all_states
        else: states = current_state

        state_skipped=False

        timestep = self.timestep.value_in_unit(unit=unit.femtoseconds)
        for k in states:            
            # progress bar
            try:
                from tqdm import tqdm
                iterator = tqdm(range(niterations), ascii=True, desc="Alchemical state %3d" % (k))
            except:
                iterator = progressbar(range(niterations),           "Alchemical state %3d" % (k), 40)
            
            if save_state:
                if self.decouple_groups_nb:
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
                ene_states=[k-1,k,k+1]  
            n_ene_states = len(ene_states)

            #Initialize dU pandas dataframe
            simulation_lambda_types, simulation_lambda_values = self.get_state_lambda(k)
            simulation_lambda_types.insert(0, 'times(ps)')
            # print(muti_idx_names)
            times_lambda_tuples = [(i*nsteps*timestep/1000,)+tuple(simulation_lambda_values)  for i in range(0, niterations)]
            muti_idx = pd.MultiIndex.from_tuples(times_lambda_tuples, names=simulation_lambda_types)
            zero_shape = np.zeros((niterations, n_ene_states))
            columns_ = [tuple(self.get_state_lambda(l)[1]) for l in range(0,n_ene_states)]
            single_simulation_df = pd.DataFrame(zero_shape, columns=columns_)
            single_simulation_df.index = muti_idx

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
                    single_simulation_df.iloc[iteration, l] = self.context.getState(getEnergy=True).getPotentialEnergy() / kT

            # save state and dU files
            if save_state: self.saveState(statefile)

            if self.decouple_groups_nb:
                csv_prefix='state_g'+str(self.current_group_nb)+'_'
            else: csv_prefix='state_'
            csv_file=csv_prefix+'s'+str(k)+'.csv'
            single_simulation_df.to_csv(csv_file)
            # np.save(dU_file, self.u_kln_array)

        #self.u_kln = self.format_u_kln(self.u_kln_array)

    @property
    def final_state(self):
        return self.context.getState(getPositions=True,getVelocities=True,enforcePeriodicBox=True)


