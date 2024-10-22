from utils.Restraints_Select import Res_atom_select, RestraintParam
import json

class REPF_para_opti():
    '''Obtain the restraint equilibrium values by reading the restraint atoms from the provided plumed.dat for the optimization of restrained degrees of freedom.
    '''
    def init(self):
        '''
        plumed_input_file: the input file for plumed reading (for candidate restraint parameters measurement);default:plumed.dat
        plumed_output_file: the plumed output file (recording the restraint atoms and parameters) ;default:Colvar
        plumed_record_freq: the frequency of plumed recording during the preliminary simulation.
        fake_state_xml:for skipping the first state simulation in alchemical simulation;default:state_s0.xml
        first_state_csv:for restraint free energy calculation;default:state_s0.csv
        '''
        plumed_input_file='plumed.dat'
        plumed_output_file='Colvar'
        plumed_record_freq=100
        fake_state_xml=self.fake_state_xml
        first_state_csv=self.first_state_csv
    
    
    def rest_para_opti(self,plumed_input_file,plumed_output_file):
        '''Define the coordinate file and topology file: the current version of program only support the coordinate file with suffix of "rst7", and the topology with suffix of "prmtop".
        ligname:default:'MOL'
        opt_cost_name:The name of the optimization method used;default:RED_E_cost
        '''
        complex_coor='./example/protein.rst7'
        complex_topo='./example/protein.prmtop'
        ligname = 'MOL'# the residue name of the ligan
        lambdas_group=json.load(open("lambdas.json"))['lambda_com_32normal']
        Restr_test =Res_atom_select(complex_coor, complex_topo, plumed_input_file, plumed_output_file, 100 )#The instantiation of the  restraint specification object.
        Restr_test.defi_rest_atoms('plumed.dat')#Read restraint atoms from plumed.dat.
        res_parm = Restr_test.aly_traj_get_best_rest(lambdas_group, 'state_s0.xml', 'state_s0.csv', 1000000,4,opt_cost_name='RED_E_cost' , if_mean=False,if_init_pose=False )
        return (res_parm)
    





























