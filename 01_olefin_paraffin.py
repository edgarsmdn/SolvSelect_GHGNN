'''
Project: Solvent_preselection

                        Olefin - paraffin 
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from solvent_preselection import screen_with_relative_volatility, get_compound, screen_with_minSF

# --- Load data
df_com = pd.read_csv('data/Pure_compound_data.csv')
df_solv = pd.read_csv('data/Molecular_solvents.csv')

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
    screen_with_relative_volatility(mixture, df_solv, AD)
    screen_with_minSF(mixture, df_solv)
