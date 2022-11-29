'''
Project: Solvent_preselection

                        Olefin - paraffin 
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from solvent_preselection import get_compound, solvent_preselection

# --- Load data
df_com = pd.read_csv('data/Pure_compound_data.csv')
df_solv = pd.read_csv('data/Molecular_solvents.csv')

solvents = df_solv['SMILES'].tolist()
solvents_names = df_solv['Name'].tolist()

mixtures = [
        {
        'c_i': get_compound('n-hexane', df_com),
        'c_j': get_compound('1-hexene', df_com),
        'mixture_type': 'olefin_paraffin',
        'T_range': (25, 74),
        },
        {
        'c_i': get_compound('n-butane', df_com),
        'c_j': get_compound('2-butene', df_com),
        'mixture_type': 'olefin_paraffin',
        'T_range': (-15, 8),
        },
        {
        'c_i': get_compound('n-heptane', df_com),
        'c_j': get_compound('1-heptene', df_com),
        'mixture_type': 'olefin_paraffin',
        'T_range': (25, 103),
        },
        {
        'c_i': get_compound('n-propane', df_com),
        'c_j': get_compound('propene', df_com),
        'mixture_type': 'olefin_paraffin',
        'T_range': (-42, -30),
        }
    ]

AD = 'both'

for mixture  in mixtures:
    sp = solvent_preselection(mixture, solvents, AD, solvents_names)
    sp.screen_with_rv()
    sp.screen_with_minSF()
