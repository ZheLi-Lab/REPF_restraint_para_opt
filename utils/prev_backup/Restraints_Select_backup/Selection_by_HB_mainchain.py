import parmed as pmd
import mdtraj as md
class Sel_rest_atm_hb_mainchain_based():
    def __init__(self, traj, parm, ligname='MOL'):
        '''Initializing
        
        Parameters
        ----------
        traj: <class 'mdtraj.core.trajectory.Trajectory'>
            The trajectory class generated by mdtraj.
        parm: such as <class 'parmed.gromacs.gromacstop.GromacsTopologyFile'>
            The topology class generated by parmed.
        ligname: str, default: 'MOL'
            The residue name of ligand.
            
        Key properties
        ----------
        self.init_atm_dict: dict
            A dict with two keys -- 'HB_MOL' and 'HB_RES', whose every value is a list containing <class 'mdtraj.core.topology.Atom'>, which are the heavy atoms that forming hydrogen bond.
        self.muti_three_res_atom_lst: list
            A list containing many list in which is like [res_CA, res_C, res_O] and store the atom index (starting from 0)
        self.group_lst: list
            A list containing <class 'Hb_pair_group'>
        self.res_lst: list
            A list containing many list. 
            Every list in the self.res_lst containing the index of the six atoms needed for restraint.(Start from 1)
        self.if_skip_aly: bool
            Judge if there is no hbond between the ligand and receptor. If so(this value equal to 1), then skip the analysis. 
        '''
        self.ligname = ligname
        self.traj = traj
        self.top = self.traj.topology
        self.parm = parm
        '''
        md.baker_hubbard identifies hydrogen bonds baced on cutoffs for the Donor-H...Acceptor distance and angle. 
        The criterion employed is  𝜃>120  and  𝑟H...Acceptor<2.5𝐴  in at least 10% of the trajectory. 
        The return value is a list of the indices of the atoms (donor, h, acceptor) that satisfy this criteria.
        '''
        hbonds = md.baker_hubbard(traj, periodic=False)
        self.protein_names=["ALA","ARG","ASH","ASN","ASP","CYM","CYS",
                            "CYX","GLH","GLN","GLU","GLY","HID","HIE",
                            "HIP","HIS","ILE","LEU","LYN","LYS","MET",
                            "PHE","PRO","SER","THR","TRP","TYR","VAL"]
        self.init_atm_dict = {'HB_MOL':[], 'HB_RES':[]}
        self.if_skip_aly = 1

        for hbond in hbonds:
            if self.ligname not in str(self.top.atom(hbond[0])) and self.ligname in str(self.top.atom(hbond[2])):
                if str(self.top.atom(hbond[0]).residue.name) in self.protein_names:
                    self.init_atm_dict['HB_MOL'].append(self.top.atom(hbond[2]))
                    self.init_atm_dict['HB_RES'].append(self.top.atom(hbond[0]))
                    self.if_skip_aly = self.if_skip_aly*0
                else:
                    '''
                    Not consider water or other non-standard residue Hbond so far.
                    '''
                    self.if_skip_aly = self.if_skip_aly*1
            elif self.ligname in str(self.top.atom(hbond[0])) and self.ligname not in str(self.top.atom(hbond[2])):
                if str(self.top.atom(hbond[2]).residue.name) in self.protein_names:
                    self.init_atm_dict['HB_MOL'].append(self.top.atom(hbond[0]))
                    self.init_atm_dict['HB_RES'].append(self.top.atom(hbond[2]))
                    self.if_skip_aly = self.if_skip_aly*0
                else:
                    '''
                    Not consider water or other non-standard residue Hbond so far.
                    '''
                    self.if_skip_aly = self.if_skip_aly*1
            else:
                '''
                Prevent define the restraint atom according to the intramolecular Hbond. 
                '''
                self.if_skip_aly = self.if_skip_aly*1
        self.muti_three_res_atom_lst = []
        self.group_lst = []
        self.res_lst = []

    def get_res_three_atm_by_HB_mainchain(self, ):
        '''Select the CA, C, O of the amino acids that form hydrogen bond with the ligand.
        
        Generated or update properties
        ----------
        self.muti_three_res_atom_lst: list
            A list containing many list in which is like [res_CA, res_C, res_O] and store the atom index (starting from 0)
        '''
        top = self.top
        muti_three_res_atom_lst = []
        for atom in self.init_atm_dict['HB_RES']:
            res_idx = atom.residue.resSeq 
            res_CA = top.select(f'residue {res_idx} and name CA')[0]
            res_C = top.select(f'residue {res_idx} and name C')[0]
            res_O = top.select(f'residue {res_idx} and name O')[0]
            single_res_list = [res_CA, res_C, res_O]
            muti_three_res_atom_lst.append(single_res_list)
        self.muti_three_res_atom_lst = muti_three_res_atom_lst

    def get_res_three_atm(self, cutoff_=0.3, frame=0):
        '''Using mdtraj.compute_neighbors to find all the amino acids within the specific cutoff from the ligand. 
        And select the CA, C, O of these amino acids. Not tested for validation. Thus not recommend!
        
        Parameters
        ----------
        cutoff_: float
            The cutoff distance, unit: nm.
        frame: int
            The frame used to analyze.
            
        Generated or update properties
        ----------
        self.muti_three_res_atom_lst: list
            A list containing many list in which is like [res_CA, res_C, res_O] and store the atom index (starting from 0)
        '''
        top = self.top
        query = top.select(f'resname {self.ligname}')
        haystack = top.select('protein')
        selected_atm_idx_lst = md.compute_neighbors(self.traj, cutoff=cutoff_, query_indices=query, haystack_indices=haystack)
        selected_res_idx = list(set([top.atom(i).residue.resSeq for i in selected_atm_idx_lst[frame]]))
        muti_three_res_atom_lst = []
        for res_ in selected_res_idx:
            res_CA = top.select(f'residue {res_} and name CA')[0]
            res_C = top.select(f'residue {res_} and name C')[0]
            res_O = top.select(f'residue {res_} and name O')[0]
            single_res_list = [res_CA, res_C, res_O]
            muti_three_res_atom_lst.append(single_res_list)
        self.muti_three_res_atom_lst = muti_three_res_atom_lst
    
    def generate_group(self):
        '''Do the self.get_res_three_atm_by_HB_mainchain() first in this function and update self.group_lst.
        
        Generated or update properties
        ----------
        self.group_lst: list
            A list containing many <class 'Hb_pair_group'>   
        '''
        # self.get_res_three_atm()
        self.get_res_three_atm_by_HB_mainchain()
        for atom in self.init_atm_dict['HB_MOL']:
            for three_res_atom_lst in self.muti_three_res_atom_lst:
                self.group_lst+=[Hb_pair_group_mainchain(self.parm, atom.index, three_res_atom_lst)]
    
    def generate_2nd(self):
        '''Do the self.generate_group() first in this function and update the 2nd and 3rd restraint atom info.
        
        '''
        self.generate_group()
        for group in self.group_lst:
            group.process()
            
    def get_final_res_lst(self):
        '''Do the self.generate_2nd() first, and update the self.res_lst.
        
        Return
        ----------
        self.res_lst: list
            A list containing many list. 
            Every list in the self.res_lst containing the index of the six atoms needed for restraint.(Start from 1)
        or 
        []: empty list, when self.if_skip_aly = 1
        '''
        if self.if_skip_aly:
            return []
        else:
            self.generate_2nd()
            for group in self.group_lst:
                for speci_restr in group.speci_restr_lst:
                    self.res_lst.extend(speci_restr.res_lst)
            return self.res_lst

class Hb_pair_group_mainchain():
    def __init__(self, parm, mol1statm: int, res_three_atm: list):
        '''Initializing
        
        Parameters
        ----------
        parm: such as <class 'parmed.gromacs.gromacstop.GromacsTopologyFile'>
            The topology class generated by parmed.
        mol1statm: int 
            The index of ligand atom in the HB bond. (Start from 0)
        res_three_atm: list
            The list contain the index of the three receptor atoms for restraint. (The index starts from 0)
        
        Key properties
        ----------
        self.speci_restr_lst: list
            A list containing many <class 'Speci_restr_mainchain'>
        '''
        self.parm = parm
        self.mol1statm = mol1statm
        self.res_three_atm = res_three_atm
        self.speci_restr_lst = []
    
    def get_2ndatm(self):
        '''Using the topological info to find the second bonded atoms for the ligand.
        
        Generated or update properties
        ----------
        self.speci_restr_lst: list
            A list containing many <class 'Speci_restr'>
        '''
        mol2ndatm_lst = [i.idx for i in self.parm.atoms[self.mol1statm].bond_partners if i.element_name != 'H']

        for mol2ndatm in mol2ndatm_lst:
            speci_restr = Speci_restr_mainchain(self.parm, self.mol1statm, mol2ndatm, self.res_three_atm[0], self.res_three_atm[1], self.res_three_atm[2])
            self.speci_restr_lst.append(speci_restr)
    
    def process(self):
        '''Do self.get_2ndatm(), and use Speci_restr_mainchain().get_3rdatm() to find possible third atom of the ligand.
        
        '''
        self.get_2ndatm()
        for speci_restr in self.speci_restr_lst:
            speci_restr.get_3rdatm()

class Speci_restr_mainchain():
    
    def __init__(self, parm, mol1statm: int, mol2ndatm: int, res1statm: int, res2ndatm: int, res3rdatm: int):
        '''Initializing
        Parameters
        ----------
        parm: such as <class 'parmed.gromacs.gromacstop.GromacsTopologyFile'>
            The topology class generated by parmed.
        mol1statm: int 
            The index of first restraint atom of ligand, which involves in forming the HB bond. (Start from 0)
        res1statm: int
            The index of first restraint atom of residue, which involves in forming the HB bond. (Start from 0)
        mol2ndatm: int 
            The index of second restraint atom of ligand. (Start from 0)
        res2ndatm: int
            The index of second restraint atom of residue. (Start from 0)
        res3rdatm: int
            The index of third restraint atom of residue. (Start from 0)
        
        Key properties
        ----------
        self.res_lst: list
            A list containing many list. 
            Every list in the self.res_lst containing the index of the six atoms needed for restraint.(Start from 1)
        '''
        self.parm = parm
        self.mol1statm = mol1statm
        self.mol2ndatm = mol2ndatm
        self.res1statm = res1statm
        self.res2ndatm = res2ndatm
        self.res3rdatm = res3rdatm
        self.res_lst = []
        
    def get_3rdatm(self):
        '''Use parmed.atom.bond_partners to find all possiblie third atoms of ligand.
        
        Generated or update properties
        ----------
        self.res_lst: list
            A list containing many list. 
            Every list in the self.res_lst containing the index of the six atoms needed for restraint.(Start from 1)
        '''  
        mol3rdatm_lst = [i.idx for i in self.parm.atoms[self.mol2ndatm].bond_partners if i.element_name != 'H' and i.idx != self.mol1statm]

        for mol_idx_3rd in mol3rdatm_lst:
            res = [self.mol1statm+1, self.mol2ndatm+1, mol_idx_3rd+1, self.res1statm+1, self.res2ndatm+1, self.res3rdatm+1]
            self.res_lst.append(res)

if __name__ == '__main__':
    parm = pmd.load_file('complex_res.top',  xyz='complex.gro')
    traj = md.load('complex.gro')
    a = Sel_rest_atm_hb_mainchain_based(traj,parm)
    lst_a = a.get_final_res_lst()
    print(lst_a)
