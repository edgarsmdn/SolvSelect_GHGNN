'''
Project: Solvent_preselection

                Plot relative volatility vs. SF
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solvent_preselection import get_compound, get_Antoine_constants, get_vapor_pressure_Antoine
from solvent_preselection import get_xs_for_SF, get_relative_volatility, get_gh_gnn_models
from solvent_preselection import get_gammas_GHGNN, margules_system
from GHGNN import GH_GNN
import os
from tqdm import tqdm

mixture_types = {
    'aliphatic_aromatic':[
        'n-hexane_benzene',
        'n-heptane_benzene',
        'n-octane_benzene',
        'n-nonane_benzene',
        'n-decane_benzene',
        ], 
    'olefin_paraffin':[
        'n-hexane_1-hexene',
        'n-butane_2-butene',
        'n-heptane_1-heptene',
        'n-propane_propene'
        ],
    'oxigenated_hydrocarbons':[
        'n-hexane_methanol',
        'n-hexane_ethanol',
        'n-hexane_n-propanol',
        'n-hexane_2-propanol',
        'n-hexane_acetone',
        'n-hexane_2-butanone'
        ]}

T_avg_mixtures = [55, 64, 77.5, 90.5, 102.5, 49.5, -3.5, 64, -36, 50, 54, 63.5, 
                  56.5, 50, 55]

SFs = np.linspace(0.01, 50, 30)[::-1]

df_com = pd.read_csv('data/Pure_compound_data.csv')
df_solv = pd.read_csv('data/Molecular_solvents.csv')

colors = ["#56ebd3", "#691b9e", "#76f014", "#dd3dca", "#42952e", "#e68dd9", 
          "#1c5872", "#b0e472", "#851657", "#2499d7", "#1b511d", "#badadd", 
          "#73350e", "#f0d446", "#4355b9", "#04a38f", "#e6262f", "#ee983a", 
          "#2524f9", "#899576"]

def get_vapor_pressures(mixture, T):
    
    c_i = get_compound(mixture.split('_')[0], df_com)
    c_j = get_compound(mixture.split('_')[1], df_com)
    
    Ai, Bi, Ci = get_Antoine_constants(c_i['smiles'], T)
    P_i = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
    
    Aj, Bj, Cj = get_Antoine_constants(c_j['smiles'], T)
    P_j = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
    
    return P_i, P_j

def get_rv_inf(mixture, solvent, T):
    c_i = get_compound(mixture.split('_')[0], df_com)
    c_j = get_compound(mixture.split('_')[1], df_com)
    
    P_i, P_j = get_vapor_pressures(mixture, T)
    
    ln_gamma_inf_i = GH_GNN(c_i['smiles'], solvent['smiles']).predict(T=T-273.15, AD=None)
    ln_gamma_inf_j = GH_GNN(c_j['smiles'], solvent['smiles']).predict(T=T-273.15, AD=None)
    
    gamma_inf_i = np.exp(ln_gamma_inf_i)
    gamma_inf_j = np.exp(ln_gamma_inf_j)
    rv_inf = get_relative_volatility(P_i, P_j, gamma_inf_i, gamma_inf_j)
    return rv_inf

def get_best_solvents(mixture):
    df = pd.read_excel('results/summary.xlsx', 
                                sheet_name=mixture)
    solvents_rv = df['rv_name'].dropna().to_list()
    solvents_sf = df['sf_name'].dropna().to_list()
    solvents = set(solvents_rv + solvents_sf)
    
    label_method, solv_comps = [], []
    for solvent in solvents:
        if solvent in solvents_rv and solvent in solvents_sf:
            label_method.append('Both')
            smiles = df[df['rv_name'] == solvent]['rv_smiles'].values[0]
        elif solvent in solvents_rv:
            label_method.append('inf-RV')
            smiles = df[df['rv_name'] == solvent]['rv_smiles'].values[0]
        elif solvent in solvents_sf:
            label_method.append('min-SF')
            smiles = df[df['sf_name'] == solvent]['sf_smiles'].values[0]
         
        solv_comps.append(
            {'name': solvent,
             'smiles':smiles
             })
    return solv_comps, label_method

def get_rv(mixture, solvent, T, SF):
    
    c_i = get_compound(mixture.split('_')[0], df_com)
    c_j = get_compound(mixture.split('_')[1], df_com)
    c_k = solvent
    
    P_i, P_j = get_vapor_pressures(mixture, T)
    
    x_i, x_j, x_k = get_xs_for_SF(SF)
    gh_gnn_models = get_gh_gnn_models(c_i, c_j, c_k)
    ln_gammas_inf = get_gammas_GHGNN(gh_gnn_models, T=T, AD=None)
    system = margules_system(*ln_gammas_inf)
    gamma_i, gamma_j, gamma_k = system.get_gammas(x_i, x_j, x_k)
    
    relative_volatilities = get_relative_volatility(P_i, P_j, gamma_i, gamma_j)
    
    return min(relative_volatilities)


folder_plots = 'results/plots'
if not os.path.exists(folder_plots):
    os.makedirs(folder_plots)

i=-1
for mix_type in mixture_types:
    print('============> ', mix_type)
    for mixture in mixture_types[mix_type]:
        print('----> ', mixture)
        i += 1
        T = T_avg_mixtures[i] + 273.15
        # Get best solvents
        solvents, label_method = get_best_solvents(mixture)
        
        fig = plt.figure(figsize=(7.5, 5.625))
        rvs = np.zeros(len(SFs)+1)
        for s, solvent in enumerate(tqdm(solvents)):
            # Calculate relative volatility at infinite dilution
            rvs[0] = get_rv_inf(mixture, solvent, T)
        
            # Calculate relative volatility
            for j, SF in enumerate(SFs):
                rvs[j+1] = get_rv(mixture, solvent, T, SF)
                
            plt.plot(SFs, rvs[1:], 'o-', alpha=0.7, label= '(' +label_method[s] + ') ---' + solvent['name'], 
                      color=colors[s], linewidth=0.7)
            plt.plot(SFs[0]+5, rvs[0], '*', alpha=0.7, color=colors[s])
                
        plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
        plt.axhline(y=3, color = 'r', linestyle = '--')
        plt.legend(fontsize=6, ncol=2)
        plt.xlabel('Solvent-to-feed ratio (SF)', fontsize=15) 
        plt.ylabel('Relative volatility', fontsize=15)
        plt.title(mixture.split('_')[0] + ' / ' + mixture.split('_')[1], fontsize=18)
        plt.tight_layout()
        plt.close(fig)
        fig.savefig(folder_plots+'/'+mixture +'.png', dpi=300, format='png')
        fig.savefig(folder_plots+'/'+mixture +'.svg', dpi=300, format='svg')
