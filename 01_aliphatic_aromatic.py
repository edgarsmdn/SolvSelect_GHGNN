'''
Project: Solvent_preselection

                        Aliphatic - aromatic 
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from solvent_preselection import screen_with_relative_volatility, get_compound, screen_with_minSF

# --- Load data
df_com = pd.read_csv('data/Pure_compound_data.csv')
df_solv = pd.read_csv('data/Molecular_solvents.csv')

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
        'T_range': (25, T_max),
        }

    screen_with_relative_volatility(mixture, df_solv, AD)
    screen_with_minSF(mixture, df_solv)
    
