'''
Project: Solvent_preselection

                        Aliphatic - aromatic 
                    
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

aliphatics = ['n-hexane', 'n-heptane', 'n-octane', 'n-nonane', 'n-decane']
T_max_lst = [85, 103, 130, 156, 180]

AD = 'both'

for T_max, aliphatic  in zip(T_max_lst, aliphatics):
    print('='*50)
    print('Aliphatic: ', aliphatic)
    mixture = {
        'c_i': get_compound(aliphatic, df_com),
        'c_j': get_compound('benzene', df_com),
        'mixture_type': 'aliphatic_aromatic',
        'T_range': (25 + 273.15, T_max + 273.15),
        }

    sp = solvent_preselection(mixture, solvents, AD, solvents_names)
    sp.screen_with_rv()
    sp.screen_with_minSF()
    
