import numpy as np
import pandas as pd 

aa_order = ['ALA',
 'ARG',
 'ASN',
 'ASP',
 'CYS',
 'GLN',
 'GLU',
 'GLY',
 'HIS',
 'ILE',
 'LEU',
 'LYS',
 'MET',
 'PHE',
 'PRO',
 'SER',
 'THR',
 'TRP',
 'TYR',
 'VAL']

aa3_aa1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
           'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
           'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
           'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

vdw_data = {"A": 0.05702, "R": 0.58946, "N": 0.22972, "D": 0.21051, "C": 0.14907,
            "E": 0.32837, "Q": 0.34861, "G": 0.00279, "H": 0.37694, "I": 0.37671,
            "L": 0.37876, "K": 0.45363, "M": 0.38872, "F": 0.55298, "P": 0.22790,
            "S": 0.09204, "T": 0.19341, "W": 0.79351, "Y": 0.61150, "V": 0.25674}

charge_index_data = {"A": 0.007187, "R": 0.043587, "N": 0.005392, "D": -0.02382, "C": -0.03661,
                     "E": 0.006802, "Q": 0.049211, "G": 0.179052, "H": -0.01069, "I": 0.021631,
                     "L": 0.051672, "K": 0.017708, "M": 0.002683, "F": 0.037552, "P": 0.239531,
                     "S": 0.004627, "T": 0.003352, "W": 0.037977, "Y": 0.023599, "V": 0.057004}


def parse_hydrophobicity(file):
    """
    Hydrophobicity file parser.

    :param file:
    :return:
    """

    scores = np.zeros(20)

    with open(file, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue

            el = line.split()

            if len(el) > 1:
                scores[aa_order.index(el[0])] = float(el[1])

    return {aa3_aa1[key]: value for key, value in zip(aa_order, scores)}


def parse_bulkiness(file):
    scores = np.zeros(20)

    with open(file, 'r') as f:
        for line in f:
            if line.startswith('\n'):
                continue

            el = line.split(':')
            if len(el) > 1:
                scores[aa_order.index(el[0].upper())] = float(el[1])

    return {aa3_aa1[key]: value for key, value in zip(aa_order, scores)}


def load_kabat(file):
    with open(file, 'r') as f:
        data = f.readlines()

    names = [x.split(', ')[0] for x in data[25:-7]]
    sequences = [[x.split(', ')[1].strip(), x.split(',')[2].strip()] for x in data[25:-7]]
    animals = [x.split(', ')[3].strip() for x in data[25:-7]]

    mask = [1 if len(x[0]) > 0 and len(x[1]) > 0 else 0 for x in sequences]

    names_mask = [x for m, x in zip(mask, names) if m == 1]
    animals_mask = [x for m, x in zip(mask, animals) if m == 1]
    sequences_mask = [x for m, x in zip(mask, sequences) if m == 1]
    seq = [x for x in sequences_mask if len(x[0]) > 100]

    seq = [[x[0].replace('-', ''), x[1].replace('-', '')] for x in seq]
    seq = [[x[0].replace('?', ''), x[1].replace('?', '')] for x in seq]

    VH_sequences = [x[1] for x in seq]
    VL_sequences = [x[0] for x in seq]

    return VL_sequences, VH_sequences, names_mask, animals_mask

def parse_csv(max_vh_length, max_vl_length, *args):
    heavy_seq_list = []
    light_seq_list = []
    org_list = []
    all_names = []
    
    heavy_seq_seen = set()
    light_seq_seen = set()
    
    for file in args:
        data = pd.read_csv(file, index_col=[0,1])
        name_list = data.index.get_level_values(0).unique()

        for name in name_list:
            h_seq = data.loc[name].loc['heavy']['sequence']
            l_seq = data.loc[name].loc['light']['sequence']
            if not isinstance(h_seq, str) or not isinstance(l_seq, str):
                continue
            if len(h_seq) > max_vh_length or len(l_seq) > max_vl_length:
                continue
            if h_seq not in heavy_seq_seen and l_seq not in light_seq_seen:
                heavy_seq_list.append(h_seq)
                light_seq_list.append(l_seq)
                heavy_seq_seen.add(h_seq)
                light_seq_seen.add(l_seq)
                org_list.append(data.loc[name].loc['light']['organism'])
            
        all_names.extend(name_list)
    
    return light_seq_list, heavy_seq_list, all_names, org_list