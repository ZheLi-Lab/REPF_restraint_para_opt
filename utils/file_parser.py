import types

'''
###Input file format:
[md_control]
temperature = 300
time_step = 0.01
num_steps = 10000

[md_system]
mass=10

[pme]
use_pme = True
pme_cutoff =12.0

###To use this class:
# Create an instance of the InputParser class
input_parser = InputParser('input.txt')

# Get the parsed data for the md_control section
md_control = input_parser.get_md_control()

# Get the parsed data for the md_system section
md_system = input_parser.get_md_system()

# Get the parsed data for the pme section
pme = input_parser.get_pme()

# Use the parsed data in your molecular dynamics program
run_simulation(md_control, md_system, pme)
'''

class InputParser:
    def __init__(self, filename):
        # Define the expected keys for each section and their default values
        self.def_sections()

        # Initialize variables to hold the parsed data
        self.data = {}
        for section, options in self.sections.items():
            self.data[section] = {}
            for option, default_value in options.items():
                self.data[section][option] = default_value

        self.current_section = None

        # Parse the input file
        with open(filename, 'r') as f:
            for line in f:
                # Remove inline comments
                line = line.split('#')[0].strip()
                if not line:  # Skip empty lines and comments
                    continue
                if line.startswith('[') and line.endswith(']'):
                    self.current_section = line[1:-1].lower()
                elif self.current_section is not None:
                    for item in line.split(','):
                        key, value = item.split('=')
                        key = key.strip().lower()
                        value = value.strip()
                        if key in self.sections[self.current_section]:
                            if value.isdigit():
                                self.data[self.current_section][key] = int(value)
                            elif value.replace('.', '').isdigit():
                                self.data[self.current_section][key] = float(value)
                            elif value.lower() == 'true':
                                self.data[self.current_section][key] = True
                            elif value.lower() == 'false':
                                self.data[self.current_section][key] = False
                            elif value.lower() == 'none':
                                self.data[self.current_section][key] = None
                            else:
                                self.data[self.current_section][key] = value.strip('"').strip("'")


        # Define the section-related methods dynamically
        for section in self.sections:
            def get_section_data(self, section=section):
                return self.data.get(section, {})
            setattr(self, f'get_{section}', types.MethodType(get_section_data, self))

    def def_sections(self):
        # Define the expected keys for each section and their default values
        self.sections = {
            # 'job_type': {
            #     'jobtype': 'normal' # if job type is normal[default], then use following options to control a single job.
            #     },            # TODO: FEP-Cascade, customized job type that easily perform FEP calculations.
            'normalmd': {
                'normalmd' : False, # whether perform normal MD
                'temperature': 298.0, 
                'timestep': 2,  #unit in fs, could be 4 fs, but it may cause unstablity of MD
                'nsteps': 1000,
                'save_traj': False
                },
            'alchemical': {
                'alchemical': False, #whether to run alchemical MD
                'mode': 'normal',
                'lambdas_json' : 'lambdas.json',
                'lambdas_group' : 'lambda_com_21normal',
                'simulation_lambdas_name': None,
                'set_rbfe_exception': False,
                'temperature': 298.0, 
                'timestep': 2,  #unit in fs, could be 4 fs, but it may cause unstablity of MD
                'nsteps': 100,
                'niterations': 5000,
                'current_group_nb': None,
                'current_group_chg': None,
                'pdbx': None,
                'save_traj': False,
                'reportstate_freq': 1000,
                'savetraj_freq': 5000,
                'input_state': None,
                'if_min_heat_density': False,
                'annihilate_electrostatics': True, # if set to False, this will decouple electrostatics instead of annihilate
                'cal_adj_ene_num': 5, # To set the number of the adjacent windows of current windows to calculate the internal energy, eg. cal_adj_ene_num=5, normally will be 11 windows energy calculated, if cal_adj_ene_num=all, calculate all windows energy.
                'kbond': 10,
                'ktheta1': 10,
                'ktheta2': 10,
                'kphi1':10,
                'kphi2':10,
                'kphi3':10,
                'alchemical_co_ion': False, # should be False or int, the residue number of the co-ion (0-based), if set to False, will not add the co-ion
                },
            'restraint': {
                'restraint': False, 
                'temperature': 298.0, 
                'timestep': 2,  #unit in fs, could be 4 fs, but it may cause unstablity of MD
                'ligand_resname': 'MOL',
                'iflog_restraint_detail': False,
                'heat_nsteps' : 25000 ,
                'heat_iterations' : 50 ,
                'density_nsteps' : 25000, # density
                'npt_nsteps' : 500000, # production with plumed
                'f_npt_state' : 'npt_final_state.xml', # npt_state file
                'lambdas_json': 'lambdas.json',
                'lambdas_group': 'lambda_com_21normal',
                'fake_state_xml': 'state_s0.xml', # fake state xml file for passing the simulation of state_s0
                'first_state_csv': 'state_s0.csv', # the ene loging file of free state
                'save_traj' : True,
                'reportstate_freq': 1000,
                'savetraj_freq': 5000,
                'f_plumed_input': 'plumed.dat',
                'f_plumed_output': 'Colvar',
                'plumed_freq': 100 ,# plumed record frequency
                'f_restraint': 'res_databystd.csv',
                'f_restraint2': 'res_databystd2.csv',
                'res_sele_strategy': 'lig_shape|HB_pair|HB_mainchain|Huggins',
                'fix_lig_3atoms_index': None, # '2126|2149|2147' should be used with 'res_sele_strategy' = 'Fix_lig_3atoms'
                'opt_cost_name': False, # could be 'RED_E_cost' or 'dG_forward' or False
                'if_mean': True, # if_mean is True, use the mean values of six parameters as the eq values for the harmonic restraint
                'if_init_pose': False,  # if_mean is True, use the initial values of six parameters from the input pose as the eq values for the harmonic restraint
                'preliminary_md_inital_state': None,
                'preliminary_min_and_heat': True,
                'crd': None, # assigned by the openmm-FEP-run.py -c option, but you can give the coordinate of the structure that you like for the generation of candidate groups of restraint
                'top': None, # assigned by the openmm-FEP-run.py -p option, but you can give the topology of the structure that you like for the generation of candidate groups of restraint
                }
        }



if __name__ == '__main__':
    def main():
        filename = 'input.txt'
        parser = InputParser(filename)

        # Test the md_control section
        md_control = parser.get_md_control()
        print('md_control section:', md_control)

        # Test the md_system section
        md_system = parser.get_md_system()

        # Test the pme section
        pme = parser.get_pme()


    main()

