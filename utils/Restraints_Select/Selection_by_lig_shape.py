import pandas as pd
import parmed as pmd
import numpy as np
import math
from numpy import linalg as LA
import mdtraj as md

class Atom():
    def __init__(self, res_indx=0, res_name='NON', atom_name='NON', atom_indx=0, coord_x=0.00, coord_y=0.00, coord_z=0.00):
        self.res_name = res_name
        self.res_indx = int(res_indx)
        self.atom_name = atom_name
        self.atom_indx = int(atom_indx)
        self.coord_x = float(coord_x)
        self.coord_y = float(coord_y)
        self.coord_z = float(coord_z)

    @property
    def get_name(self):
        return self.atom_name
    
    @property
    def get_res_name(self):
        return self.res_name

    @property
    def get_coord(self):
        return self.coord_x, self.coord_y, self.coord_z

    @property
    def get_atom_indx(self):
        return self.atom_indx
    
    @property
    def get_res_indx(self):
        return self.res_indx

    def set_name(self, atom_name):
        self.atom_name = atom_name

    def set_xyz(self, coord_x, coord_y, coord_z):
        self.coord_x = float(coord_x)
        self.coord_y = float(coord_y)
        self.coord_z = float(coord_z)

    def set_atom_indx(self, atom_indx):
        self.atom_indx = int(atom_indx)


    def calc_dist(self, atom2):
        '''Calculate the distance between the atom itself and another atom.

        Parameter
        ----------
        atom2: <class 'Atom'>

        Return
        ----------
        dist_sqr ** 0.5: The distance.
        '''
        dist_sqr = (self.coord_x - atom2.coord_x)**2 + (self.coord_y - atom2.coord_y)**2 + (self.coord_z - atom2.coord_z)**2
        return dist_sqr ** 0.5
    
    def calc_dist_xyz(self, coord):
        '''Calculate the distance between the atom itself and another atom by giving the coordinate of another atom.

        Parameter
        ----------
        coord: array_like
            A array_like object that stores three floats, which are x, y, z.

        Return
        ----------
        dist_sqr ** 0.5: The distance.
        '''
        dist_sqr = (self.coord_x - coord[0])**2 + (self.coord_y - coord[1])**2 + (self.coord_z - coord[2])**2
        return dist_sqr ** 0.5
    
    def calc_angle(self, atom2, atom3):
        '''Calculate the angle between the atom itself, atom2, and atom3.

        Parameter
        ----------
        atom2: <class 'Atom'>
        atom3: <class 'Atom'>

        Return
        ----------
        theta/math.pi*180.0: The angle in degree.

        '''
        np.seterr(divide='ignore', invalid='ignore')
        vec1=np.array([ self.coord_x - atom2.coord_x,  self.coord_y - atom2.coord_y,  self.coord_z - atom2.coord_z ])
        vec2=np.array([atom3.coord_x - atom2.coord_x, atom3.coord_y - atom2.coord_y, atom3.coord_z - atom2.coord_z ])
        cos_theta=vec1.dot(vec2)/(LA.norm(vec1)*LA.norm(vec2))
        try:
            theta=math.acos(cos_theta)
        except:
            theta=0
        return theta/math.pi*180.0
        
    def calc_dihedral(self, atom2, atom3, atom4):
        '''Calculate the dihedral between the atom itself, atom2, atom3 and atom4.

        Parameter
        ----------
        atom2: <class 'Atom'>
        atom3: <class 'Atom'>
        atom4: <class 'Atom'>

        Return
        ----------
        theta/math.pi*180.0: The dihedral in degree.
        '''
        np.seterr(divide='ignore', invalid='ignore')
        vec1=np.array([atom2.coord_x -  self.coord_x, atom2.coord_y -  self.coord_y, atom2.coord_z -  self.coord_z ])
        vec2=np.array([atom3.coord_x - atom2.coord_x, atom3.coord_y - atom2.coord_y, atom3.coord_z - atom2.coord_z ])
        vec3=np.array([atom4.coord_x - atom3.coord_x, atom4.coord_y - atom3.coord_y, atom4.coord_z - atom3.coord_z ])
        fa1=np.cross(vec1, vec2)
        fa2=np.cross(vec2, vec3)
        cos_theta=fa1.dot(fa2)/(LA.norm(fa1)*LA.norm(fa2))
        theta=math.acos(cos_theta)
        return theta/math.pi*180.0        
    
    def __str__(self):
        return '%13s%7.3f%10.3f%10.3f%-2s' % (self.coord_x,self.coord_y,self.coord_z)

    def __repr__(self):
        return "Atom('" + '%-2s %-2s' % (self.atom_indx,self.atom_name) + "')"
        
class Residue():
    def __init__(self):
        self.atom_list=[]

    def add_atom(self, atom):
        self.atom_list.append(atom)
        
 
    
class Ligand(Residue):
    def find_longest_atoms(self):
        '''Find the two most distant non-hydrogen atoms within a ligand.

        Generated or update properties
        ----------
        self.atom_longest1: <class 'Atom'>
        self.atom_longest2: <class 'Atom'>
        '''
        self.dist_longest=0.0
        for atom1 in self.atom_list:
            for atom2 in self.atom_list:
                if self.dist_longest < Atom.calc_dist(atom1,atom2):
                    if atom1.get_name[0] != "H" and atom2.get_name[0] != "H":
                        self.dist_longest = Atom.calc_dist(atom1,atom2) 
                        self.atom_longest1 = atom1
                        self.atom_longest2 = atom2

                
class Protein(Residue):
    @property
    def get_center(self):
        i=0
        sum_x=sum_y=sum_z=0.0
        for atom in self.atom_list:
            i+=1
            sum_x+=atom.coord_x
            sum_y+=atom.coord_y
            sum_z+=atom.coord_z           
        cent_x=sum_x/i
        cent_y=sum_y/i
        cent_z=sum_z/i
        return cent_x, cent_y, cent_z



class Frame():
    def __init__(self):
        '''Initializing
        
        Key properties
        ----------
        self.atom_list: list
            The list contain many <class 'Atom'>
        self.protein_names: list
            The list contain many string of the standard residue name. 
            ["ALA","ARG","ASH","ASN","ASP","CYM","CYS",
             "CYX","GLH","GLN","GLU","GLY","HID","HIE",
             "HIP","HIS","ILE","LEU","LYN","LYS","MET",
             "PHE","PRO","SER","THR","TRP","TYR","VAL"]
        '''
        self.atom_list=[]
        self.protein_names=["ALA","ARG","ASH","ASN","ASP","CYM","CYS",
                            "CYX","GLH","GLN","GLU","GLY","HID","HIE",
                            "HIP","HIS","ILE","LEU","LYN","LYS","MET",
                            "PHE","PRO","SER","THR","TRP","TYR","VAL"]

    def add_atom(self, atom):
        self.atom_list.append(atom)
        
    def initiate(self, ligname='MOL'):
        '''Initializing the properties like self.protein and self.ligand. And do find the two most distant non-hydrogen atoms within a ligand.

        Generated or update properties
        ----------
        self.protein: <class 'Protein'>
        self.ligand: <class 'Ligand'>
        '''
        self.protein=Protein()
        self.ligand=Ligand()
        for atom in self.atom_list:
            if atom.get_res_name in self.protein_names:
                self.protein.add_atom(atom)
            elif atom.get_res_name == ligname:
                self.ligand.add_atom(atom)
        self.ligand.find_longest_atoms() 

    def get_lig_longest_cent_atom(self):
        '''For a pair of heavy atoms that are farthest from each other in the ligand small molecule, the one that is farthest from the center of mass of the ligand will be selected as the first atom from the ligand, which will be used in restraint definition.

        Return
        ----------
        self.restrain_ligatom_1: <class 'Atom'>
            The first atom from the ligand, which will be used in restraint definition.
        '''
        if not self.ligand:
            print ("Warning: <class 'Frame'> not initiated, will initiate first")
            self.initiate()
        dist1 = self.ligand.atom_longest1.calc_dist_xyz(self.protein.get_center)
        dist2 = self.ligand.atom_longest2.calc_dist_xyz(self.protein.get_center)
        self.restrain_ligatom_1 = self.ligand.atom_longest1 if dist1 < dist2 else self.ligand.atom_longest2
        return self.restrain_ligatom_1
    
    def get_nearest_atom(self,coord,): 
        '''The heavy atoms on the small molecule ligands are scanned, and the heavy atom closest to the provided coordinate will be returned.
        Parameter
        ----------
        coord: array_like
            A array_like object that stores three floats, which are x, y, z.

        Return
        ----------
        atom_near: <class 'Atom'>
            The the heavy atom closest to the provided coordinate.
        '''
        dist_near = 99.0
        for atom in self.ligand.atom_list:
            #print (repr(atom))
            if atom.get_name[0] != "H":
                dist = atom.calc_dist_xyz(coord)
                #print (dist)
                if dist < dist_near:
                    atom_near = atom
                    dist_near = dist
                    #print (atom)
                    #print (dist)
        return atom_near   
    
    def get_lig_cent_atom(self):
        '''The heavy atoms on the small molecule ligands are scanned, and the heavy atom closest to the midpoint of the furthest atomic pair is determined as the second selected atom from the ligand.

        Return
        ----------
        self.restrain_ligatom_2: <class 'Atom'>
            The second atom from the ligand, which will be used in restraint definition.
        '''
        ligatm1_xyz=np.array(self.ligand.atom_longest1.get_coord)
        ligatm2_xyz=np.array(self.ligand.atom_longest2.get_coord)
        self.restrain_ligatom_2 = self.get_nearest_atom((ligatm1_xyz + ligatm2_xyz)/2)     
        return self.restrain_ligatom_2
    
    def get_lig_3rd_atom(self):
        '''Based on the first ligand restraint atom is obtained by self.get_lig_longest_cent_atom(), and the second atom is got by self.get_lig_cent_atom(), the program will select the heavy atom of the small molecule as the third atom that satisfies the following two conditions. First, the angle between the atom and the second and first selected heavy atom is between 45 degrees and 135 degrees; second, the heavy atom is the farthest away from the second selected atom in the small molecule heavy atoms.

        Return
        ----------
        self.restrain_ligatom_3: <class 'Atom'>
            The third atom from the ligand, which will be used in restraint definition.
        '''
        dist_far = 0.0
        for atom in self.ligand.atom_list:
            if atom.get_name[0] != "H":
                angle = atom.calc_angle(self.restrain_ligatom_2,self.restrain_ligatom_1)
                dist = atom.calc_dist(self.restrain_ligatom_2)
                if 45.0 < angle < 135.0 and dist > dist_far:
                    dist_far = dist
                    atom_far = atom
        self.restrain_ligatom_3 = atom_far
        return self.restrain_ligatom_3

    def get_lig_3rd_atom_byatom1(self,atom1):
        '''Based on the first atom is given by the parameter "atom1" and the second atom is got by self.get_lig_cent_atom(), the program will select the heavy atom of the small molecule as the third atom that satisfies the following two conditions. First, the angle between the atom and the second and first selected heavy atom is between 45 degrees and 135 degrees; second, the heavy atom is the farthest away from the second selected atom in the small molecule heavy atoms.
        
        Parameter
        ----------
        atom1: <class 'Atom'>

        Return
        ----------
        self.restrain_ligatom_3: <class 'Atom'>
            The third atom from the ligand, which will be used in restraint definition.
        '''
        dist_far = 0.0
        for atom in self.ligand.atom_list:
            if atom.get_name[0] != "H":
                angle = atom.calc_angle(self.restrain_ligatom_2,atom1)
                dist = atom.calc_dist(self.restrain_ligatom_2)
                if 45.0 < angle < 135.0 and dist > dist_far:
                    dist_far = dist
                    atom_far = atom
        self.restrain_ligatom_3 = atom_far
        return self.restrain_ligatom_3
    
    def get_nearest_CA_atom(self):
        '''Select the CA atom closest to the first bound atom from the ligand. If the first bound atom of the ligand has not been generated, use self.get_lig_longest_cent_atom() to generate it.

        Return
        ----------
        self.restrain_protatom_1 <class 'Atom'>
            The first atom from the protein, which will be used in restraint definition.
        '''
        if not self.restrain_ligatom_1:
            print ("Warning: should get restrain_ligatom_1 first")
            self.get_lig_longest_cent_atom()
        self.nearest_dist=99.0
        for atom in self.protein.atom_list:
            if atom.get_name == "CA":
                dist = self.restrain_ligatom_1.calc_dist(atom)
                if dist < self.nearest_dist:
                    self.restrain_protatom_1 = atom
                    self.nearest_dist = dist
        return self.restrain_protatom_1
    
    def get_nearest_CO_atom(self):
        '''According to self.restrain_protatom_1, the C and O atoms of the corresponding amino acids are selected as the second and third constraint atoms from the protein.

        Return
        ----------
        self.restrain_protatom_2, self.restrain_protatom_3: tuple
            A tuple containing two <class 'Atom'>, which are the second and third atom from the protein, which will be used in restraint definition.
        '''
        for atom in self.protein.atom_list:
            if atom.res_indx == self.restrain_protatom_1.res_indx:
                if atom.atom_name == "C":
                    self.restrain_protatom_2 = atom
                if atom.atom_name == "O":
                    self.restrain_protatom_3 = atom
        return self.restrain_protatom_2, self.restrain_protatom_3


def get_res_idx_within_one_residue(traj, speci_res, cutoff_=0.3, frame=0, ):
    '''Using mdtraj.compute_neighbors to find all the amino acids within the specific cutoff from the specific residue.
    
    Parameters
    ----------
    traj: <class 'mdtraj.core.trajectory.Trajectory'>
        The trajectory class generated by mdtraj.
    speci_res: str
        The name of the specific residue, which may not be the standard amino acid.
    cutoff_: float
        The cutoff distance, unit: nm.
    frame: int
        The frame used to analyze.

    Return 
    ----------
    selected_res_idx: list
        A list contains the index (Start from 0) of the residue selected by this function.
    '''
    top = traj.topology
    query = top.select(f'resname {speci_res}')
    haystack = top.select('protein')
    selected_atm_idx_lst = md.compute_neighbors(traj, cutoff=cutoff_, query_indices=query, haystack_indices=haystack)
    selected_res_idx = list(set([top.atom(i).residue.resSeq for i in selected_atm_idx_lst[frame]]))
    return selected_res_idx

def ligand_shape_based_sel(traj, lig_resi_name='MOL'):
    '''LIGAND'S RESTRAINT ATOMS: Any one of a pair of heavy atoms with the farthest distance from each other in the ligand small molecule is used as the first restraint atom from the ligand, and then the heavy atom closest to the midpoint of the above-mentioned farthest atomic pair is selected as the second atom from the ligand. select the heavy atom of the small molecule as the third atom that satisfies the following two conditions. First, the angle between the atom and the second and first selected heavy atom is between 45 degrees and 135 degrees; second, the heavy atom is the farthest away from the second selected atom in the small molecule heavy atoms.
RECEPTOR'S RESTRAINT ATOMS: The CA, C, and O atoms of amino acids within three angstroms from the ligand serve as the first, second, and third bound atoms from the acceptor, respectively.

    Parameters
    ----------
    traj: <class 'mdtraj.core.trajectory.Trajectory'>
        The trajectory class generated by mdtraj.
    lig_resi_name: str
        The residue name of the ligand, default: 'MOL'.

    Return
    ----------
    lst_z: list
        A list containing many list. 
        Every list in the lst_z containing the index of the six atoms needed for restraint.(Start from 1)

    Example
    ----------
    >>> import numpy as np
    >>> import math
    >>> from numpy import linalg as LA
    >>> import mdtraj as md
    >>> traj = md.load('complex.gro')
    >>> ligname = 'MOL'
    >>> lst_z = ligand_shape_based_sel(traj, ligname)
    '''
    frame = Frame()
    top = traj.topology
    for i in top.atoms:
        xyz = traj.xyz[0][i.index]
        atom_x = xyz[0]
        atom_y = xyz[1]
        atom_z = xyz[2]
    #     print(atom_x)
        residue_name = i.residue.name
        residue_id = i.residue.resSeq
        atom_name = i.name
        atom_id = i.index+1
        frame.add_atom(Atom(residue_id,residue_name,atom_name,atom_id,atom_x,atom_y,atom_z))
    frame.initiate(ligname=lig_resi_name)
    atom_lig11=frame.ligand.atom_longest1
    atom_lig12=frame.get_lig_cent_atom()
    atom_lig13=frame.get_lig_3rd_atom_byatom1(atom_lig11)

    atom_lig21=frame.ligand.atom_longest2
    atom_lig22=frame.get_lig_cent_atom()
    atom_lig23=frame.get_lig_3rd_atom_byatom1(atom_lig21)
    lig_atoms_1=[int(atom_lig11.get_atom_indx), int(atom_lig12.get_atom_indx), int(atom_lig13.get_atom_indx)]
    lig_atoms_2=[int(atom_lig21.get_atom_indx), int(atom_lig22.get_atom_indx), int(atom_lig23.get_atom_indx)]
    lig_atoms = [lig_atoms_1, lig_atoms_2]
    lst_z=[]
    ligname=lig_resi_name
    within3A_res_list = get_res_idx_within_one_residue(traj, ligname, 0.3, 0)
    # print(within3A_res_list)
    #selection base on resname
    muti_three_res_atom_lst = []
    for i in within3A_res_list:
        res_CA = top.select(f'residue {i} and name CA')[0]+1
        res_C = top.select(f'residue {i} and name C')[0]+1
        res_O = top.select(f'residue {i} and name O')[0]+1
        single_res_list = [res_CA, res_C, res_O]
        muti_three_res_atom_lst.append(single_res_list)
        for lig_atom in lig_atoms:
            single_six = lig_atom+single_res_list
            lst_z.append(single_six)
    return lst_z 




if __name__ == '__main__':
    import pandas as pd
    import parmed as pmd
    import numpy as np
    import math
    from numpy import linalg as LA
    import mdtraj as md
    traj = md.load('complex.gro')
    ligname = 'MOL'
    iflog=True
    lst_z = ligand_shape_based_sel(self.traj, lig_resi_name)
    if iflog:
        grp_num1=len(lst_z)
        print(f"Selected by first_stategy: {lst_z} with number of {grp_num1}")
        lst_z_file=open('based_lig_shape', 'w+')
        print(str(lst_z), file=lst_z_file)
        lst_z_file.close()

