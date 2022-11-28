'''
Project: Solvent_preselection

                        Collect results
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd

n_best = 10

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

df_results = pd.DataFrame({'Ranking':range(1,n_best+1)})

with pd.ExcelWriter('results/summary.xlsx') as writer:
    for mix_type in mixture_types:
        for mixture in mixture_types[mix_type]:
            best_rv = pd.read_csv('results/'+mix_type+'/best_solvents/'+mixture+'_best.csv')
            best_sf = pd.read_csv('results/'+mix_type+'/best_solvents/'+mixture+'_best_minSF.csv')
            
            best_rv = best_rv[best_rv['mean'].notna()]
            best_sf = best_sf[best_sf['mean'].notna()]
            
            df_results['rv_name'] =  best_rv['Solvent_name'].iloc[:n_best]
            df_results['sf_name'] =  best_sf['Solvent_name'].iloc[:n_best]
            
            df_results['rv_smiles'] =  best_rv['Solvent_SMILES'].iloc[:n_best]
            df_results['sf_smiles'] =  best_sf['Solvent_SMILES'].iloc[:n_best]
            
            df_results.to_excel(writer, sheet_name=mixture, index=False)
            
        
        
        


