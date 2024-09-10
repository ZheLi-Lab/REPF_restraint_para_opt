from operator import ge
import numpy as np

class Atom():
    def __init__(self, atom_info):
        info_list=atom_info.strip().split()
        len_info=len(info_list)
        if len_info >= 8 and (info_list[0] == 'ATOM' or info_list[0] == 'HETATM'):
            
            self.atomid = int(info_list[1])
            self.atomname = info_list[2]
            self.resname = info_list[3]
            self.resid = int(info_list[4])
            self.coord = [float(crd) for crd in info_list[5:8]]
            self.charge_info = [float(chg) for chg in info_list[8:-1]]
            self.group_num = info_list[-1]
        else:
            raise ValueError("Atom information not correct: %s" % (atom_info))
    @property
    def atomtype(self):
        return ''.join([i for i in self.atomname if not i.isdigit()])
    
    def update_chg_info(self, new_chg, col):
        '''
        col is the index of elements in atom.charge_info, starting from 1.
        '''
        assert isinstance(new_chg, (float,np.float64)), "new_chg must be float"
        self.charge_info[col-1] = new_chg

    def update_group_num(self, new_group_num,):
        assert isinstance(new_group_num, int), "new_group_num must be int"
        self.group_num = new_group_num

    def write_atm_line(self):
        empty_altLoc = ' '
        empty_chainID = ' '
        empty_iCode = ' '
        formatted_charge_ = ["{: >10.6f}".format(chg) for chg in self.charge_info]
        charge_info_str = " ".join(formatted_charge_)
        nb_group_idx = self.group_num
        # print(charge_info_str)
        l_str = '{: <6s}{: >5d} {: ^4s}{}{:3s} {}{:4d}{}   {:8.3f}{:8.3f}{:8.3f} {} {}\n'.format(
            'HETATM',
            self.atomid,
            self.atomname,
            empty_altLoc,
            self.resname,
            empty_chainID,
            self.resid,
            empty_iCode,
            self.coord[0],
            self.coord[1],
            self.coord[2],
            charge_info_str,
            nb_group_idx,
        )
        return l_str

    def __repr__(self):
        return "Atom('" + ':%-2s@%-2s' % (self.resname,self.atomname) + "')"
    def __str__(self):
        return self.atomname
    



class PdbxParser():
    def __init__(self, pdbx_file):
        self.pdbx = self.loadPDBX(pdbx_file)
        self.atoms_list=[]
        for line in self.pdbx:
            try:
                atom=Atom(line)
                self.atoms_list.append(atom)
            except ValueError:
                pass

    def loadPDBX(self, file):
        if isinstance(file, str):
            with open(file, 'r') as f:
                pdbx = f.readlines()
        else:
            pdbx = file.readlines()
        return pdbx
    
    def writePDBX(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            for atom in self.atoms_list:
                f.write(atom.write_atm_line()) 

    def get_group_nb_dict(self,start_id):
        group_nb={}
        atom_id=0
        for atom in self.atoms_list:
            group_name='group'+str(atom.group_num)
            if group_name in group_nb:
                group_nb[group_name].add(atom_id+start_id)
            else:
                group_nb[group_name]={atom_id+start_id} # create this set
            atom_id += 1
        return group_nb

    def get_charge_list(self,start_id,col=1):
        #get the initial_charge_dict and target_charge_dict based on the pdbx file.
        #start_id means the first atom id of the alchemical region
        #col (starts from 1) means which charge column will be considered as the initial charges. 
        #if the col is set to 0, then the initial charge will be used as the target charge
        #        the next charge column from the initial charge column will be considered as the target charges.
        #        if there is no target charges column, target charges will be set to 0.0.
        initial_charge_dict={}
        target_charge_dict={}
        atom_id=0
        if col == 0:
                initial_col=col
                target_col=col
        else:
                initial_col=col-1
                target_col=col
        for atom in self.atoms_list:
            initial_charge_dict[atom_id+start_id]=atom.charge_info[initial_col]
            try:
                target_charge_dict[atom_id+start_id]=atom.charge_info[target_col]
            except:
                target_charge_dict[atom_id+start_id]=0.0
            atom_id += 1
        return initial_charge_dict, target_charge_dict

    def get_net_chg_of_atms_by_atmid_lst(self,atomid_list,col):
        '''
        atomid_list is a list of atomids read from the pdbx file
        col is the index of elements in atom.charge_info, starting from 1.
        '''
        net_chg = 0
        for atomid in atomid_list:
            for atom in self.atoms_list:
                if atom.atomid == atomid:
                    net_chg += atom.charge_info[col-1]
        return net_chg
    
    def find_positive_chg_atms(self, atomid_list, which_chg_col):
        '''
        atomid_list is a list of atomids read from the pdbx file;
        which_chg_col is the index of elements in atom.charge_info, starting from 1.
        '''
        posi_chg_atms = []
        for atm in self.atoms_list:
            if atm.charge_info[which_chg_col-1] > 0 and atm.atomid in atomid_list:
                posi_chg_atms.append(atm)
        return posi_chg_atms
    
    def find_negative_chg_atms(self, atomid_list, which_chg_col):
        '''
        atomid_list is a list of atomids read from the pdbx file;
        which_chg_col is the index of elements in atom.charge_info, starting from 1.
        '''
        nega_chg_atms = []
        for atm in self.atoms_list:
            if atm.charge_info[which_chg_col-1] < 0 and atm.atomid in atomid_list:
                nega_chg_atms.append(atm)
        return nega_chg_atms
    
    def update_atom_prop(self, atomid, prop_name, new_value, **kwargs):
        '''
        atomid is read from the pdbx file;
        prop_name is the name of the property to update;
        **kwargs is the a dictionary of additional keyword arguments, like the 'which_chg_col'(starting from 1) keyword to assign which element in the atom.charge_info need to change;
        '''
        for atm in self.atoms_list:
            if atm.atomid == atomid:
                if 'which_chg_col' in kwargs.keys():
                    old_chg_info = getattr(atm, prop_name)
                    old_chg_info[kwargs['which_chg_col']-1] = new_value
                    
                    setattr(atm, prop_name, old_chg_info)
                else:
                    setattr(atm, prop_name, new_value)
      
    def get_atom_prop(self, atomid, prop_name):
        for atm in self.atoms_list:
            if atm.atomid == atomid:
                prop = getattr(atm, prop_name)
        return prop
    
    def dup_last_chg_info_col(self, ):
        for atm in self.atoms_list:
            atm.charge_info.append(atm.charge_info[-1])

    def annihilate_grps(self, grp_dict, dechg_strategy='balanced'):
        '''
        grp_dict is a dictionary: key like 'group_1', value like an atomid_list [atomid_1, atomid_2, ...]
        '''
        start_grp_id = 1
        for key, value in grp_dict.items():
            if dechg_strategy == 'balanced':
                self.balanced_dechg_one_grp(value, start_grp_id)
            for atom in self.atoms_list:
                if atom.atomid in value:
                    atom.group_num = start_grp_id
            start_grp_id+=1

    def dechg_all_devdw_bygroups(self, grp_dict):
        '''
        decharge all the atom simultaneously, and decouple the vdw term by groups.
        grp_dict is a dictionary: key like 'group_1', value like an atomid_list [atomid_1, atomid_2, ...]
        '''
        start_grp_id = 1
        for key, value in grp_dict.items():
            for atom in self.atoms_list:
                if atom.atomid in value:
                    atom.group_num = start_grp_id
                    atom.charge_info = [0.0, 0.0]
            start_grp_id += 1
    
    def dechg_bygroup_devdw_bygroups(self, grp_dict):
        '''
        decharge all the atom simultaneously, and decouple the vdw term by groups.
        grp_dict is a dictionary: key like 'group_1', value like an atomid_list [atomid_1, atomid_2, ...]
        '''
        start_grp_id = 1
        for key, value in grp_dict.items():
            for atom in self.atoms_list:
                if atom.atomid in value:
                    atom.group_num = start_grp_id
                    example_atom_chg_info = self.get_atom_prop(self.atoms_list[0].atomid, 'charge_info')
                    if len(example_atom_chg_info) == start_grp_id:
                        self.dup_last_chg_info_col()
                    if atom.group_num == 1:    
                        self.update_atom_prop(atom.atomid, 'charge_info', 0.0000, which_chg_col=start_grp_id+1)
            start_grp_id += 1
    

    def balanced_dechg_one_grp(self, atomid_list, bef_chg_info_col, ): 
        '''
        atomid_list is a list of atomids read from the pdbx file;
        bef_chg_info_col is the index of charge to be changed in atom.charge_info, starting from 1;
        aft_chg_info_col is the index of charge that have changed in atom.charge_info, starting from 1, aft_chg_info_col must be sum of bef_chg_info_col and 1.
        '''
        aft_chg_info_col = bef_chg_info_col+1
        example_atom_chg_info = self.get_atom_prop(self.atoms_list[0].atomid, 'charge_info')
        if len(example_atom_chg_info) == bef_chg_info_col:
            self.dup_last_chg_info_col()
        before_change_grp_netchg = np.around(self.get_net_chg_of_atms_by_atmid_lst(atomid_list, bef_chg_info_col), decimals=6)
        print(f'before_change_grp_netchg: {before_change_grp_netchg}')
        # set the decharge group's aft_chg_info
        for atomid in atomid_list:
            self.update_atom_prop(atomid, 'charge_info', 0.0000, which_chg_col=aft_chg_info_col)
        # get the remaining non-zero-charge atomid_list
        remain_non_zero_atomid_list = []
        for atm in self.atoms_list:
            # first check if the atom is in the atomid_list that we want to annihilate
            if atm.atomid not in atomid_list:
                # second check if the atom's bef_chg_info is non-zero
                if atm.charge_info[bef_chg_info_col] != 0.0:
                    remain_non_zero_atomid_list.append(atm.atomid)
        # now assign the before_change_grp_netchg to the remaining non-zero-charge atom
        # if the remain_non_zero_atomid_list is empty list, then skip the assignment
        if len(remain_non_zero_atomid_list) == 0:
            pass
        else:
            if before_change_grp_netchg < 0:
                positive_chg_atms = self.find_positive_chg_atms(remain_non_zero_atomid_list, bef_chg_info_col)
                positive_chg_atmsid_list = [ atm.atomid for atm in positive_chg_atms ]
                net_posi_chg = np.around(self.get_net_chg_of_atms_by_atmid_lst(positive_chg_atmsid_list, bef_chg_info_col), decimals=6)
                for atomid in positive_chg_atmsid_list:
                    # weighted assignment
                    old_chg = self.get_atom_prop(atomid, 'charge_info')[bef_chg_info_col-1]
                    new_chg = np.around(old_chg + before_change_grp_netchg*old_chg/net_posi_chg, decimals=6)
                    self.update_atom_prop(atomid, 'charge_info', new_chg, which_chg_col=aft_chg_info_col)
            else:
                negative_chg_atms = self.find_negative_chg_atms(remain_non_zero_atomid_list, bef_chg_info_col)
                negative_chg_atmsid_list = [ atm.atomid for atm in negative_chg_atms ]
                net_nega_chg = np.around(self.get_net_chg_of_atms_by_atmid_lst(negative_chg_atmsid_list, bef_chg_info_col), decimals=6)
                for atomid in negative_chg_atmsid_list:
                    old_chg = np.around(self.get_atom_prop(atomid, 'charge_info')[bef_chg_info_col-1], decimals=6)
                    new_chg = np.around(old_chg + before_change_grp_netchg*old_chg/net_nega_chg, decimals=6)
                    self.update_atom_prop(atomid, 'charge_info', new_chg, which_chg_col=aft_chg_info_col)
        all_atom_ids_list = [ atom.atomid for atom in self.atoms_list ] 
        after_dechg_sys_netchg = self.get_net_chg_of_atms_by_atmid_lst(all_atom_ids_list, aft_chg_info_col)
        print(f'After balanced charge assignment, the net charge of the ligand is {after_dechg_sys_netchg}')
        
                
                
                
        

if __name__ == "__main__":
    pdbx=PdbxParser('lig.pdbx')
    group_nb = pdbx.get_group_nb_dict(41)
    initial_charge, target_charge = pdbx.get_charge_list(41,col=1)
    print(group_nb,initial_charge,target_charge)

