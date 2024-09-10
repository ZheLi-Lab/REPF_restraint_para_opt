import re
import numpy as np
class Metainteraction():
    def __init__(self, resnr, restype, reschain, resnr_lig, restype_lig, reschain_lig, lig_coor, prot_coor):
        self.resnr = resnr
        self.restype = restype
        self.reschain = reschain
        self.resnr_lig = resnr_lig
        self.restype_lig = restype_lig
        self.reschain_lig = reschain_lig
        self.lig_coor = lig_coor
        self.prot_coor = prot_coor
        self.lig_coor_x = float(self.lig_coor.strip().split(',')[0])
        self.lig_coor_y = float(self.lig_coor.strip().split(',')[1])
        self.lig_coor_z = float(self.lig_coor.strip().split(',')[2])
        self.lig_coors = np.array([self.lig_coor_x, self.lig_coor_y, self.lig_coor_z])
        self.prot_coor_x = float(self.prot_coor.strip().split(',')[0])
        self.prot_coor_y = float(self.prot_coor.strip().split(',')[1])
        self.prot_coor_z = float(self.prot_coor.strip().split(',')[2])
        self.prot_coors = np.array([self.prot_coor_x, self.prot_coor_y, self.prot_coor_z])

class HydrophobicInteraction(Metainteraction):
    def __init__(self, resnr, restype, reschain, resnr_lig, restype_lig, reschain_lig,
                 dist, lig_carbon_idx, prot_carbon_idx, lig_coor, prot_coor):
        super().__init__(resnr, restype, reschain, resnr_lig, restype_lig, reschain_lig, lig_coor, prot_coor)
        self.dist = dist
        self.lig_carbon_idx = lig_carbon_idx
        self.prot_carbon_idx = prot_carbon_idx

    def __repr__(self):
        return f"HydrophobicInteraction(resnr={self.resnr}, restype='{self.restype}', reschain='{self.reschain}', resnr_lig={self.resnr_lig}, restype_lig='{self.restype_lig}', reschain_lig='{self.reschain_lig}', dist={self.dist}, lig_carbon_idx={self.lig_carbon_idx}, prot_carbon_idx={self.prot_carbon_idx}, lig_coor='{self.lig_coor}', prot_coor='{self.prot_coor}')"

class HydrogenBond(Metainteraction):
    def __init__(self, resnr, restype, reschain, resnr_lig, restype_lig, reschain_lig, 
                 sidechain, dist_H_A, dist_D_A, don_angle, protisdon, donor_idx, donor_type,
                 acceptor_idx, acceptor_type, lig_coor, prot_coor):
        super().__init__(resnr, restype, reschain, resnr_lig, restype_lig, reschain_lig, lig_coor, prot_coor)
        self.sidechain = sidechain
        self.dist_H_A = dist_H_A
        self.dist_D_A = dist_D_A
        self.don_angle = don_angle
        self.protisdon = protisdon
        self.donor_idx = donor_idx
        self.donor_type = donor_type
        self.acceptor_idx = acceptor_idx
        self.acceptor_type = acceptor_type

    def __repr__(self):
        return f"HydrogenBond(resnr={self.resnr}, restype='{self.restype}', reschain='{self.reschain}', resnr_lig={self.resnr_lig}, restype_lig='{self.restype_lig}', reschain_lig='{self.reschain_lig}', sidechain={self.sidechain}, dist_H_A={self.dist_H_A}, dist_D_A={self.dist_D_A}, don_angle={self.don_angle}, protisdon={self.protisdon}, donor_idx={self.donor_idx}, donor_type='{self.donor_type}', acceptor_idx={self.acceptor_idx}, acceptor_type='{self.acceptor_type}', lig_coor='{self.lig_coor}', prot_coor='{self.prot_coor}')"

class WaterBridge(Metainteraction):
    def __init__(self, resnr, restype, reschain, resnr_lig, restype_lig, reschain_lig, dist_A_W, dist_D_W, don_angle, water_angle, protisdon, 
                 donor_idx, donor_type, acceptor_idx, acceptor_type, water_idx, lig_coor, prot_coor, water_coor):
        super().__init__(resnr, restype, reschain, resnr_lig, restype_lig, reschain_lig, lig_coor, prot_coor)
        self.dist_A_W = dist_A_W
        self.dist_D_W = dist_D_W
        self.don_angle = don_angle
        self.water_angle = water_angle
        self.protisdon = protisdon
        self.donor_idx = donor_idx
        self.donor_type = donor_type
        self.acceptor_idx = acceptor_idx
        self.acceptor_type = acceptor_type
        self.water_idx = water_idx
        self.water_coor = water_coor
        self.water_coor_x = float(self.water_coor.strip().split(',')[0])
        self.water_coor_y = float(self.water_coor.strip().split(',')[1])
        self.water_coor_z = float(self.water_coor.strip().split(',')[2])
        

    def __repr__(self):
        return f"WaterBridge(resnr={self.resnr}, restype='{self.restype}', reschain='{self.reschain}', resnr_lig={self.resnr_lig}, restype_lig='{self.restype_lig}', reschain_lig='{self.reschain_lig}', dist_A_W={self.dist_A_W}, dist_D_W={self.dist_D_W}, don_angle={self.don_angle}, water_angle={self.water_angle}, protisdon={self.protisdon}, donor_idx={self.donor_idx}, donor_type='{self.donor_type}', acceptor_idx={self.acceptor_idx}, acceptor_type='{self.acceptor_type}', water_idx={self.water_idx}, lig_coor='{self.lig_coor}', prot_coor='{self.prot_coor}', water_coor='{self.water_coor}')"



# 初始化一个Interaction 对象来保存一个残基相关的相互作用组内所有的 Metainteraction 对象
class Interaction:
    def __init__(self, title, ):
        self.title = title
        match = re.search(r'(.+):[A-Z]:(\d+)', self.title)
        if match:
            interaction_residue_name = match.group(1)  
            interaction_residue_num = match.group(2)   
        else:
            print("Not match!")
        self.interaction_residue_name = interaction_residue_name
        self.interaction_residue_num = interaction_residue_num
        self.hydrophobic_interaction_coll = []
        self.hydrogenbond_interactions_coll = []
        self.waterbridge_interaction_coll = []
    
    def __repr__(self):
        return f'''Interaction residue name: {self.interaction_residue_name}
Interaction residue number: {self.interaction_residue_num}
Hydrophobic interaction collection: {self.hydrophobic_interaction_coll}
HydrogenBond interaction collection: {self.hydrogenbond_interactions_coll}
Waterbridge interaction collection: {self.waterbridge_interaction_coll}
'''
    
def parse_Plip_report(report_file_path):
    # 用于匹配起始行的正则表达式模式
    start_pattern = re.compile(r'(.+):[A-Z]:\d+ \(.+\) - SMALLMOLECULE')

    # 打开文件并读取内容
    with open(report_file_path, 'r') as file:
        lines = file.readlines()

    # 初始化变量以保存每个组的内容
    all_groups = []
    current_group = []

    # 遍历文件的每一行
    for line in lines:
        if start_pattern.search(line):
            # 如果匹配到起始行，将之前的组添加到all_groups中
            if current_group:
                all_groups.append(current_group)
            # 开始记录新组的内容
            current_group = [line]
        elif current_group:
            # 如果当前行不是起始行且当前组已经开始记录，将当前行添加到当前组
            current_group.append(line)
    # 确保最后一组也被添加到all_groups中
    if current_group:
        all_groups.append(current_group)
    return all_groups

def parse_sing_res_interaction(single_group_list):
#     print('111111')
#     print(single_group_list)
    current_group = []  # 用于存储当前组的行

    ts_Interaction = Interaction(single_group_list[0].strip())

    for line in single_group_list:
        current_group.append(line)
        if line.startswith('|'):
            # 如果行以 '|' 开头，表示表格数据行
            columns = line.strip('|').strip().split('|')
            columns = [col.strip() for col in columns if col.strip()]
            if len(columns) == 11:
                # 如果表格有 11 列，表示是 Hydrophobic Interaction
                try:
                    interaction = HydrophobicInteraction(*columns)
                    ts_Interaction.hydrophobic_interaction_coll.append(interaction)
                except:
                    pass
#                     print(columns)
            elif len(columns) == 17:
                # 如果表格有 17 列，表示是 Hydrogen Bond
                try:
                    interaction = HydrogenBond(*columns)
                    ts_Interaction.hydrogenbond_interactions_coll.append(interaction)
                except:
                    pass
#                     print(columns)
            elif len(columns) == 19:
                try:
                # 如果表格有 19 列，表示是 Water Bridge
                    interaction = WaterBridge(*columns)
                    ts_Interaction.waterbridge_interaction_coll.append(interaction)
                except:
                    pass
#                     print(columns)

    return ts_Interaction

def parse_plip_report_main(report_file_path):
    all_groups = parse_Plip_report(report_file_path)
    for single_group in all_groups:
        #MOL:A:299 (MOL) - SMALLMOLECULE
        if single_group[0].strip().split(':')[0] == 'MOL':
#             print(single_group)
            mol_interaction = parse_sing_res_interaction(single_group)
#         else:
#             print(single_group[0].strip().split(':')[0])
    return mol_interaction

