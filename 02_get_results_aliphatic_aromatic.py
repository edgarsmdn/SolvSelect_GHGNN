'''
Project: Solvent_preselection

                        Aliphatic - aromatic 
                            Analyze results
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import matplotlib.pyplot as plt
import pandas as pd
import os

aliphatics = ['n-hexane', 'n-heptane', 'n-octane', 'n-nonane', 'n-decane']
mixture_type = 'aliphatic_aromatic'

tanimoto_threshold = 0.35     
class_threshold = 25         
n_best_solvents = 50
n_best_solvents_plot = 5

df_brouwer = pd.read_csv('data/Molecular_solvents_brouwer_best.csv')
df_brouwer = df_brouwer[df_brouwer[mixture_type] == 'Yes']

folder = 'results/'+mixture_type
folder_best = folder+'/best_solvents'
if not os.path.exists(folder_best):
    os.makedirs(folder_best)
    
colors = ["#69ef7b", "#6f2b6e", "#b8e27d", "#074d65", "#8ed4c6"]

for aliphatic in aliphatics:
    print('='*70)
    print(aliphatic)
    df = pd.read_csv(folder+'/'+aliphatic + '_benzene.csv')
    Ts = df.iloc[:,2:].columns.tolist()
    df_ad = pd.read_csv(folder+'/'+aliphatic + '_benzene_AD_both.csv')
    
    df['mean'] = df.iloc[:,2:].mean(axis=1)
    df['Feasible_tanimoto'] = (df_ad['max_10_sim_i'] >= tanimoto_threshold) & (df_ad['max_10_sim_j'] >= tanimoto_threshold)
    df['Feasible_class'] = (df_ad['n_class_i'] >= class_threshold) & (df_ad['n_class_j'] >= class_threshold)
    df['Feasible'] = (df['Feasible_tanimoto']) & (df['Feasible_class']) #|
    
    # --- Minimum SF
    
    df_SF = pd.read_csv(folder+'/'+aliphatic + '_benzene_minSF.csv')
    df_SF['mean'] = df_SF.iloc[:,2:].mean(axis=1)
    df_feasible_SF = df_SF[df['Feasible'] == True]
    df_feasible_SF = df_feasible_SF.sort_values(by=['mean'], ascending=True)
    df_feasible_SF.to_csv(folder_best+'/'+aliphatic + '_benzene_best_minSF.csv', index=False)
    df_feasible_SF_notna = df_feasible_SF[df_feasible_SF['mean'].notna()]
    print('Feasible prediction solvents SF: ', df_feasible_SF_notna.shape[0])
    print(' ')
    for sm, solvent in zip(df_feasible_SF_notna['Solvent_SMILES'].iloc[:n_best_solvents], 
                           df_feasible_SF_notna['Solvent_name'].iloc[:n_best_solvents]):
        if sm in df_brouwer['SMILES'].tolist():
            brouwer_name = df_brouwer[df_brouwer['SMILES'] == sm]['Name'].values[0]
            message = '{:<40}  {:>15} '.format(brouwer_name, sm)
            print(message)
    
    # Plot
    df_best = df_feasible_SF_notna.iloc[:n_best_solvents_plot]
    
    fig = plt.figure(figsize=(7.5, 5.625))
    for s in range(df_best.shape[0]):
        x = [float(T) for T in Ts]
        y = df_best.iloc[s].loc[Ts]
        
        plt.plot(x,y, 'o-', alpha=0.7, label=df_best['Solvent_name'].iloc[s], 
                  color=colors[s], linewidth=0.5)
    plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
    plt.legend(fontsize=6)
    plt.xlabel('Temperature (°C)', fontsize=15) 
    plt.ylabel('minimum solvent-to-feed ratio', fontsize=15)
    plt.title(aliphatic + ' / benzene', fontsize=18)
    plt.tight_layout()
    plt.close(fig)
    fig.savefig(folder_best+'/top_5_' +aliphatic +'_benzene_minSF.png', dpi=300, format='png')
    # fig.savefig(folder_best+'/parity.svg', dpi=300, format='svg')
    
    print('-'*50)
    
    # --- Relative volatility at infinite dilution
    
    df_feasible = df[df['Feasible'] == True]
    df_feasible = df_feasible.sort_values(by=['mean'], ascending=False)
    df_feasible.to_csv(folder_best+'/'+aliphatic + '_benzene_best.csv', index=False)
    print('Feasible prediction solvents: ', df_feasible.shape[0])
    print(' ')
    print((df_feasible['mean'].iloc[:n_best_solvents] > 3).value_counts())
    print(' ')
    for sm, solvent in zip(df_feasible['Solvent_SMILES'].iloc[:n_best_solvents], 
                           df_feasible['Solvent_name'].iloc[:n_best_solvents]):
        if sm in df_brouwer['SMILES'].tolist():
            brouwer_name = df_brouwer[df_brouwer['SMILES'] == sm]['Name'].values[0]
            message = '{:<40}  {:>15} '.format(brouwer_name, sm)
            print(message)
            
        
    # Plot
    df_best = df_feasible.iloc[:n_best_solvents_plot]
    
    fig = plt.figure(figsize=(7.5, 5.625))
    for s in range(df_best.shape[0]):
        x = [float(T) for T in Ts]
        y = df_best.iloc[s].loc[Ts]
        
        plt.plot(x,y, 'o-', alpha=0.7, label=df_best['Solvent_name'].iloc[s], 
                  color=colors[s], linewidth=0.5)
    plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
    plt.legend(fontsize=6)
    plt.xlabel('Temperature (°C)', fontsize=15) 
    plt.ylabel('Relative volatility at infinite dilution', fontsize=15)
    plt.title(aliphatic + ' / benzene', fontsize=18)
    plt.tight_layout()
    plt.close(fig)
    fig.savefig(folder_best+'/top_5_' +aliphatic +'_benzene.png', dpi=300, format='png')
    # fig.savefig(folder_best+'/parity.svg', dpi=300, format='svg')
    
    