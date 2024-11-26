from utils.Restraints_Select import Res_atom_select, RestraintParam
import json

class REPF_para_opti():
    '''Obtain the restraint equilibrium values by reading the restraint atoms from the provided plumed.dat for the optimization of restrained degrees of freedom.
    '''
    def init(self,plumed_input_file,plumed_output_file):
        '''
        plumed_input_file: the input file for plumed reading (for candidate restraint parameters measurement);default:plumed.dat
        plumed_output_file: the plumed output file (recording the restraint atoms and parameters) ;default:Colvar
        '''
        self.plumed_input_file='plumed.dat'
        self.plumed_output_file='Colvar'
    
    
    def rest_para_opti(self,plumed_input_file,plumed_output_file):
        '''Define the coordinate file and topology file: the current version of program only support the coordinate file with suffix of "rst7", and the topology with suffix of "prmtop".
        ligname:default:'MOL'
        opt_cost_name:The name of the optimization method used;default:RED_E_cost
        '''

        ligname = 'MOL'# the residue name of the ligan
        lambdas_group=json.load(open("lambdas.json"))['lambda_com_32normal']
        Restr_test =Res_atom_select( plumed_input_file, plumed_output_file )#The instantiation of the  restraint specification object.
        Restr_test.defi_rest_atoms('plumed.dat')#Read restraint atoms from plumed.dat.
        res_parm = Restr_test.aly_traj_get_best_rest(lambdas_group,opt_cost_name='RED_E_cost' , if_mean=False,if_init_pose=False )
        return (res_parm)
    





























