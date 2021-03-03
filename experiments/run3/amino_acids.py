d_1aa = {'SER': 'S', 'LYS': 'K', 'PRO': 'P', 'ALA': 'A', 'ASP': 'D', 'ARG': 'R', 'VAL': 'V', 'CYS': 'C', 'HIS': 'H',
         'ASX': 'B', 'PHE': 'F', 'MET': 'M', 'LEU': 'L', 'ASN': 'N', 'TYR': 'Y', 'ILE': 'I', 'GLN': 'Q', 'THR': 'T',
         'GLX': 'Z', 'GLY': 'G', 'TRP': 'W', 'GLU': 'E'}

d_3aa = {'A': 'ALA', 'C': 'CYS', 'B': 'ASX', 'E': 'GLU', 'D': 'ASP', 'G': 'GLY', 'F': 'PHE', 'I': 'ILE', 'H': 'HIS',
         'K': 'LYS', 'M': 'MET', 'L': 'LEU', 'N': 'ASN', 'Q': 'GLN', 'P': 'PRO', 'S': 'SER', 'R': 'ARG', 'T': 'THR',
         'W': 'TRP', 'V': 'VAL', 'Y': 'TYR', 'Z': 'GLX'}

dihed_count = {'CYS': 1, 'ASP': 2, 'SER': 1, 'GLN': 3, 'LYS': 4, 'ILE': 2, 'PRO': 1, 'THR': 1, 'PHE': 2, 'ALA': 0,
               'GLY': 0, 'HIS': 2, 'GLU': 3, 'LEU': 2, 'ARG': 4, 'TRP': 2, 'VAL': 1, 'ASN': 2, 'TYR': 2, 'MET': 3}

# Identified manually using my personal judgement # TODO: add backbone oxygens
FUNCTIONAL_GROUPS = {'hydroxyl': {'SER': [[['OG'], ['HG']]],
                                  'THR': [[['OG1'], ['HG1']]],
                                  'TYR': [[['OH'], ['HH']]]},

                     'carboxyl': {'ASP': [[['CG'], ['OD1'], ['OD2'], ['HD2']]],
                                  'GLU': [[['CG'], ['OE1'], ['OE2'], ['HE2']]]},

                     'amide': {'ASN': [[['CG'], ['OD1'], ['ND2'], ['2HD2', 'HD22'], ['1HD2', 'HD21']]],
                               'GLN': [[['CD'], ['OE1'], ['NE2'], ['2HE2', 'HE22'], ['1HE2', 'HE21']]]},

                     'methyl': {'ALA': [[['CB'], ['1HB', 'HB1'], ['2HB', 'HB2'], ['3HB', 'HB3']]],
                                'VAL': [[['CG1'], ['1HG1', 'HG11'], ['2HG1', 'HG12'], ['3HG1', 'HG13']],
                                        [['CG2'], ['1HG2', 'HG21'], ['2HG2', 'HG22'], ['3HG2', 'HG23']]],
                                'ILE': [[['CD1'], ['1HD1', 'HD11'], ['2HD1', 'HD12'], ['3HD1', 'HD13']],
                                        [['CG2'], ['1HG2', 'HG21'], ['2HG2', 'HG22'], ['3HG2', 'HG23']]],
                                'LEU': [[['CD1'], ['1HD1', 'HD11'], ['2HD1', 'HD12'], ['3HD1', 'HD13']],
                                        [['CD2'], ['1HD2', 'HD21'], ['2HD2', 'HD22'], ['3HD2', 'HD23']]],
                                'THR': [[['CG2'], ['1HG2', 'HG21'], ['2HG2', 'HG22'], ['3HG2', 'HG23']]]},

                     'aromatic': {'PHE': [[['CG'], ['CD1'], ['CE1'], ['CZ'], ['CE2'], ['CD2'], ['HD1'], ['HE1'], ['HZ'], ['HE2'], ['HD2']]],
                                  'TYR': [[['CG'], ['CD1'], ['CE1'], ['CZ'], ['CE2'], ['CD2'], ['HD1'], ['HE1'], ['HE2'], ['HD2']]],
                                  'TRP': [[['CG'], ['CD1'], ['HD1'], ['CD2'], ['NE1'], ['HE1'], ['CE2'], ['CE3'], ['HE3'], ['CZ2'], ['HZ2'], ['CZ3'], ['HZ3'], ['CH2'], ['HH2']]],
                                  'HIS': [[['CG'], ['ND1'], ['HD1'], ['CE1'], ['HE1'], ['NE2'], ['HE2'], ['CD2'], ['HD2']]]},

                     'positive': {'ARG': [[['NE'], ['HE'], ['CZ'], ['NH1'], ['1HH1', 'HH11'], ['2HH1', 'HH12'], ['NH2'], ['1HH2', 'HH21'], ['2HH2', 'HH22']]],
                                  'HIS': [[['CG'], ['ND1'], ['HD1'], ['CE1'], ['HE1'], ['NE2'], ['HE2'], ['CD2'], ['HD2']]],
                                  'LYS': [[['NZ'], ['1HZ', 'HZ1'], ['2HZ', 'HZ2'], ['3HZ', 'HZ3']]]},

                     'negative': {'ASP': [[['CG'], ['OD1'], ['OD2'], ['HD2']]],
                                  'GLU': [[['CG'], ['OE1'], ['OE2'], ['HE2']]]}}

residue_bonds_noh = {
    'GLY': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C')},
    'ALA': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'), 'CB': ('CA',)},
    'CYS': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'SG'), 'SG': ('CB',)},
    'SER': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'OG'), 'OG': ('CB',)},
    'MET': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'SD'), 'SD': ('CG', 'CE'), 'CE': ('SD',)},
    'LYS': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'CD'), 'CD': ('CG', 'CE'), 'CE': ('CD', 'NZ'), 'NZ': ('CE',)},
    'ARG': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'CD'), 'CD': ('CG', 'NE'), 'NE': ('CD', 'CZ'), 'CZ': ('NE', 'NH1', 'NH2'),
            'NH1': ('CZ',), 'NH2': ('CZ',)},
    'GLU': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'CD'), 'CD': ('CG', 'OE1', 'OE2'), 'OE1': ('CD',), 'OE2': ('CD',)},
    'GLN': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'CD'), 'CD': ('CG', 'OE1', 'NE2'), 'OE1': ('CD',), 'NE2': ('CD',)},
    'ASP': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'OD1', 'OD2'), 'OD1': ('CG',), 'OD2': ('CG',)},
    'ASN': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'OD1', 'ND2'), 'OD1': ('CG',), 'ND2': ('CG',)},
    'LEU': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'CD1', 'CD2'), 'CD1': ('CG',), 'CD2': ('CG',)},
    'HIS': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'ND1', 'CD2'), 'ND1': ('CG', 'CE1'), 'CD2': ('CG', 'NE2'),
            'CE1': ('ND1', 'NE2'), 'NE2': ('CD2', 'CE1')},
    'PHE': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'CD1', 'CD2'), 'CD1': ('CG', 'CE1'), 'CD2': ('CG', 'CE2'),
            'CE1': ('CD1', 'CZ'), 'CE2': ('CD2', 'CZ'), 'CZ': ('CE1', 'CE2')},
    'TYR': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'CD1', 'CD2'), 'CD1': ('CG', 'CE1'), 'CD2': ('CG', 'CE2'),
            'CE1': ('CD1', 'CZ'), 'CE2': ('CD2', 'CZ'), 'CZ': ('CE1', 'CE2', 'OH'), 'OH': ('CZ',)},
    'TRP': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'CD1', 'CD2'), 'CD1': ('CG', 'NE1'), 'CD2': ('CG', 'CE2', 'CE3'),
            'NE1': ('CD1', 'CE2'), 'CE2': ('CD2', 'NE1', 'CZ2'), 'CE3': ('CD2', 'CZ3'), 'CZ3': ('CE3', 'CH2'),
            'CZ2': ('CE2', 'CH2'), 'CH2': ('CZ2', 'CZ3')},
    'VAL': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG1', 'CG2'), 'CG1': ('CB',), 'CG2': ('CB',)},
    'THR': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'OG1', 'CG2'), 'OG1': ('CB',), 'CG2': ('CB',)},
    'ILE': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA',), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG1', 'CG2'), 'CG1': ('CB', 'CD1'), 'CD1': ('CG1',), 'CG2': ('CB',)},
    'PRO': {'OXT': ('C',), 'C': ('CA', 'O', 'OXT'), 'O': ('C',), 'N': ('CA', 'CD'), 'CA': ('N', 'C', 'CB'),
            'CB': ('CA', 'CG'), 'CG': ('CB', 'CD'), 'CD': ('N', 'CG')},
}
