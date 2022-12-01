'''
Project: Solvent_preselection

        Relevant functions for solvent pre-selection
                    using the GH-GNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.optimize import minimize_scalar
import warnings
import ast
from GHGNN import GH_GNN
import os

_df_antoine = pd.read_csv('data/Antoine_constants.csv')

def get_selectivity(gamma_inf_i:float, gamma_inf_j:float):
    '''
    Computes the selectivity from actvity coefficients of key components

    Parameters
    ----------
    gamma_inf_i : float
        Activity coefficient at infinite dilution of solute i in the solvent.
    gamma_inf_j : float
        Activity coefficient at infinite dilution of solute j in the solvent.

    Returns
    -------
    s_ij : float
        Selectivity of compound i over compound j.

    '''
    s_ij = gamma_inf_i/gamma_inf_j
    return s_ij

def get_compound(name:str, df_comp):
    df = df_comp[df_comp['name'] == name]
    c={}
    for col in df.columns:
        if col == 'UNIFAC_Do groups':
            c[col] = ast.literal_eval(df[col].values[0])
        else:
            c[col] = df[col].values[0]
    return c

def get_Antoine_constants(smiles:str, T:float, db=_df_antoine, verbose=True):
    '''
    Parameters
    ----------
    smiles : str
        SMILES of the pure compound.
    T : float
        Temperature in K.
    db : pd.DataFrame (optional)
        Database containing the Antoine constants

    Raises
    ------
    Exception
        If the SMILES is not in the database.

    '''
    df = db[db['smiles'] == smiles]
    if len(df)== 1:
        # Check temperature range
        T1, T2 = df['T1'].values[0], df['T2'].values[0]
        if T<T1 or T>T2 and verbose:
            warnings.warn(f'Temperature of {smiles} is outside of Antoine constants range')
        return df['A'].values[0], df['B'].values[0], df['C'].values[0]
    elif len(df) > 1:
        df_T = df[(df['T1'] <= T) & (df['T2'] >= T)]
        if len(df_T) == 1:
            return df_T['A'].values[0], df_T['B'].values[0], df_T['C'].values[0]
        elif len(df_T) > 1:
            idxmax = df_T['Delta T'].idxmax() # Largest T range
            df_T_largest = df_T.loc[idxmax]
            return df_T_largest['A'], df_T_largest['B'], df_T_largest['C']
        else:
            if verbose:
                warnings.warn(f'Temperature of {smiles} is outside of Antoine constants range')
            T1_dist = np.abs(df['T1'].to_numpy() - T).reshape(-1,1)
            T2_dist = np.abs(df['T2'].to_numpy() - T).reshape(-1,1)
            T_dist_min = np.min(np.concatenate((T1_dist,T2_dist), axis=1), axis=1)
            idxclosest = np.argmin(T_dist_min)
            df_T_closest = df.iloc[idxclosest]
            return df_T_closest['A'], df_T_closest['B'], df_T_closest['C']         
    else:
        raise Exception(f'This SMILES {smiles} is not in database')

def get_vapor_pressure_Antoine(A:float, B:float, C:float, T:float):
    '''
    Computes the vapor pressure of pure compound using the Anotine equation

    Parameters
    ----------
    A : float
        Antoine constant A.
    B : float
        Antoine constant B.
    C : float
        Antoine constant C.
    T : float
        Temperature in K.

    Returns
    -------
    P : float
        Pressure in bar.

    '''
    log_P = A - B/(T + C)
    P = 10**log_P
    return P

def get_relative_volatility(P_i:float, P_j:float, gamma_inf_i:float, 
                            gamma_inf_j:float):
    '''
    Computes the relative volatility of a system considering ideal vapor phase
    and non-ideal liquid phase. This assumption is true at low pressures. In 
    case a solvent is present, this is called pseudo-binary relative volatility.
    
    A relative volatility above ~1.1 could be separated using normal distillation.
    In such cases, the key components' boiling point usually differ by more than 
    50 K.

    Parameters
    ----------
    P_i : float
        Vapor pressure of pure species i.
    P_j : float
        Vapor pressure of pure species j.
    gamma_inf_i : float
        Activity coefficient at infinite dilution of solute i in the solvent.
    gamma_inf_j : float
        Activity coefficient at infinite dilution of solute j in the solvent.

    Returns
    -------
    a_ij : float
        Relative volatility of compound i over compound j.

    '''
    a_ideal = P_i/P_j                                # Ideal part
    s_ij = get_selectivity(gamma_inf_i,gamma_inf_j)  # Non-ideal part or selectivity
    a_ij = a_ideal * s_ij
    return a_ij

def get_gh_gnn_models(c_1, c_2, c_3):
    '''
    Returns a list of all GH-GNN models corresponding to all binary systems
    of a ternary mixture (two key components and solvent)

    Parameters
    ----------
    c_1 : dict
        Key component i as dict.
    c_2 : dict
        Key component j as dict.
    c_3 : dict
        Solvent as dict.

    Returns
    -------
    gh_gnn_models : list
        All GH-GNN models for every binary system of the ternary mixture.

    '''
    gh_gnn_12 = GH_GNN(c_1['smiles'], c_2['smiles'])
    gh_gnn_21 = GH_GNN(c_2['smiles'], c_1['smiles'])
    gh_gnn_13 = GH_GNN(c_1['smiles'], c_3['smiles'])
    gh_gnn_31 = GH_GNN(c_3['smiles'], c_1['smiles'])
    gh_gnn_23 = GH_GNN(c_2['smiles'], c_3['smiles'])
    gh_gnn_32 = GH_GNN(c_3['smiles'], c_2['smiles'])

    gh_gnn_models = [gh_gnn_12, gh_gnn_21, gh_gnn_13, gh_gnn_31, gh_gnn_23, gh_gnn_32]
    return gh_gnn_models

def get_gammas_GHGNN(gh_gnn_models, T, AD):
    ln_gs = []
    for gh_gnn in gh_gnn_models:
        ln_gs.append(gh_gnn.predict(T=T, AD=AD))
    return ln_gs

class margules_system():
    def __init__(self, ln_g_12:float, ln_g_21:float, 
                       ln_g_13:float, ln_g_31:float,
                       ln_g_23:float, ln_g_32:float):
        
        self.A_12 = ln_g_12
        self.A_21 = ln_g_21
        
        self.A_13 = ln_g_13
        self.A_31 = ln_g_31
        
        self.A_23 = ln_g_23
        self.A_32 = ln_g_32
        
        A_123 = 0
        
        self.B_123 = 0.5*(self.A_12 + self.A_21 + self.A_13 + self.A_31 + 
                          self.A_23 + self.A_32) - A_123
        
    def get_gammas(self, x1, x2, x3):
        
        part_12 = x1*x2*(x2*self.A_12 + x1*self.A_21)
        part_13 = x1*x3*(x3*self.A_13 + x1*self.A_31)
        part_23 = x2*x3*(x3*self.A_23 + x2*self.A_32)
        
        GE = part_12 + part_13 + part_23 + x1*x2*x3*self.B_123
        
        gamma_1 = np.exp(
            (
                2*(x1*x2*self.A_21 + x1*x3*self.A_31) + x2**2*self.A_12 + 
                x3**2*self.A_13 + x2*x3*self.B_123 - 2*GE
            )
            )
        
        gamma_2 = np.exp(
            (
                2*(x2*x3*self.A_32 + x2*x1*self.A_12) + x3**2*self.A_23 + 
                x1**2*self.A_21 + x3*x1*self.B_123 - 2*GE
            )
            )
        
        gamma_3 = np.exp(
            (
                2*(x3*x1*self.A_13 + x3*x2*self.A_23) + x1**2*self.A_31 + 
                x2**2*self.A_32 + x1*x2*self.B_123 - 2*GE
            )
            )
        
        return gamma_1, gamma_2, gamma_3

def get_xs_for_SF(SF, n_points=30):
    
    x_k = np.repeat(SF/(1+SF), n_points)
    x_i = np.linspace(0,1-x_k[0],n_points)
    x_j = 1- x_k - x_i
    
    return x_i, x_j, x_k

class solvent_preselection():
    
    def __init__(self, mixture, solvents, AD, solvents_names=None):
        self.mixture = mixture
        self.solvents = solvents
        self.AD = AD
        if solvents_names is None:
            self.solvents_names = ['']*len(solvents)
        else:
            self.solvents_names = solvents_names
        
        # Create results folder
        self.results_folder = 'results/'
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        
    def screen_with_rv(self, n_Ts=10):
        
        mixture = self.mixture
        solvents= self.solvents
        AD = self.AD
        solvents_names = self.solvents_names
        
        # Extract mixture information
        c_i = mixture['c_i']
        c_j = mixture['c_j']
        mixture_type = mixture['mixture_type']
        T_range = mixture['T_range']
        
        mixture = c_i['name'] + '_' + c_j['name'] # Mixture name
        
        # Temperatures to be evaluated
        Ts = np.linspace(T_range[0], T_range[1], n_Ts)
        
        # Create folder to store results of screening
        if not os.path.exists(self.results_folder + mixture_type):
            os.makedirs(self.results_folder + mixture_type)

        relative_volatilities_inf = np.zeros((len(solvents), len(Ts)))
        for t, T in enumerate(Ts):
            print('---> Temperature (K): ', T)
            
            # Compute vapor pressures
            Ai, Bi, Ci = get_Antoine_constants(c_i['smiles'], T)
            P_i = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
            
            Aj, Bj, Cj = get_Antoine_constants(c_j['smiles'], T)
            P_j = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
            
            if AD is not None:
                print('\n ------ > Computing AD')
                ad_file = self.results_folder + mixture_type + '/' + mixture + '_AD_' + AD + '.csv'
                if os.path.exists(ad_file):
                    print(' ------ > AD already exists!\n')
                    AD = None
                else:
                    n_class_i_storage = np.zeros(len(solvents))
                    n_class_j_storage = np.zeros(len(solvents))
                    max_10_sim_i_storage = np.zeros(len(solvents))
                    max_10_sim_j_storage = np.zeros(len(solvents))
            
            for s, solvent in enumerate(tqdm(solvents)):
                if AD == 'both':
                    ln_gamma_inf_i, feasible_sys_i, n_class_i_storage[s], max_10_sim_i_storage[s] = GH_GNN(c_i['smiles'], solvent).predict(T=T, AD=AD)
                    ln_gamma_inf_j, feasible_sys_j, n_class_j_storage[s], max_10_sim_j_storage[s] = GH_GNN(c_j['smiles'], solvent).predict(T=T, AD=AD) 
                elif AD is None:
                    ln_gamma_inf_i = GH_GNN(c_i['smiles'], solvent).predict(T=T, AD=AD)
                    ln_gamma_inf_j = GH_GNN(c_j['smiles'], solvent).predict(T=T, AD=AD)
                else:
                    raise Exception('Current implementation only supports AD="both" or AD=None')
                gamma_inf_i = np.exp(ln_gamma_inf_i)
                gamma_inf_j = np.exp(ln_gamma_inf_j)
                a_inf = get_relative_volatility(P_i, P_j, gamma_inf_i, gamma_inf_j)
                relative_volatilities_inf[s, t] = a_inf
                
            if AD is not None:
                df_AD = pd.DataFrame({
                    'Solvent_SMILES':solvents,
                    'Solvent_name':solvents_names,
                    'n_class_i': n_class_i_storage,
                    'n_class_j': n_class_j_storage,
                    'max_10_sim_i': max_10_sim_i_storage,
                    'max_10_sim_j': max_10_sim_j_storage,
                    })
                df_AD.to_csv(ad_file, index=False)
                print(' ------ > AD computed!\n')

        df_results = pd.DataFrame(relative_volatilities_inf, columns=Ts)
        df_results.insert(0, 'Solvent_SMILES', solvents)
        df_results.insert(1, 'Solvent_name', solvents_names)
        df_results.to_csv(self.results_folder+mixture_type+'/'+mixture+'_rv.csv', index=False)

    def screen_with_minSF(self, n_Ts=10, rv_threshold=3):
        
        mixture = self.mixture
        solvents= self.solvents
        AD = self.AD
        solvents_names = self.solvents_names
        
        # Extract mixture information
        c_i = mixture['c_i']
        c_j = mixture['c_j']
        mixture_type = mixture['mixture_type']
        T_range = mixture['T_range']
        
        mixture = c_i['name'] + '_' + c_j['name'] # Mixture name
        
        # Temperatures to be evaluated
        Ts = np.linspace(T_range[0], T_range[1], n_Ts)
        
        # Create folder to store results of screening
        if not os.path.exists(self.results_folder + mixture_type):
            os.makedirs(self.results_folder + mixture_type)
            
        minSFs = np.zeros((len(solvents), len(Ts)))
        for t, T in enumerate(Ts):
            print('---> Temperature (K): ', T)
            
            # Compute vapor pressures
            Ai, Bi, Ci = get_Antoine_constants(c_i['smiles'], T)
            P_i = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
            
            Aj, Bj, Cj = get_Antoine_constants(c_j['smiles'], T)
            P_j = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
            
            if AD is not None:
                print('\n ------ > Computing AD')
                ad_file = self.results_folder + mixture_type + '/' + mixture + '_AD_' + AD + '.csv'
                if os.path.exists(ad_file):
                    print(' ------ > AD already exists!\n')
                    AD = None
                else:
                    n_class_i_storage = np.zeros(len(solvents))
                    n_class_j_storage = np.zeros(len(solvents))
                    max_10_sim_i_storage = np.zeros(len(solvents))
                    max_10_sim_j_storage = np.zeros(len(solvents))
            
            for s, solvent in enumerate(tqdm(solvents)):
                if AD == 'both':
                    ln_gamma_inf_i, feasible_sys_i, n_class_i_storage[s], max_10_sim_i_storage[s] = GH_GNN(c_i['smiles'], solvent).predict(T=T, AD=AD)
                    ln_gamma_inf_j, feasible_sys_j, n_class_j_storage[s], max_10_sim_j_storage[s] = GH_GNN(c_j['smiles'], solvent).predict(T=T, AD=AD) 
                elif AD is None:
                    ln_gamma_inf_i = GH_GNN(c_i['smiles'], solvent).predict(T=T, AD=AD)
                    ln_gamma_inf_j = GH_GNN(c_j['smiles'], solvent).predict(T=T, AD=AD)
                else:
                    raise Exception('Current implementation only supports AD="both" or AD=None')
                gamma_inf_i = np.exp(ln_gamma_inf_i)
                gamma_inf_j = np.exp(ln_gamma_inf_j)
                a_inf = get_relative_volatility(P_i, P_j, gamma_inf_i, gamma_inf_j)
                
                if a_inf < rv_threshold:
                     minSFs[s,t] = np.nan
                else:
                    c_k = {'smiles': solvent}
                    gh_gnn_models = get_gh_gnn_models(c_i, c_j, c_k)
                    ln_gammas_inf = get_gammas_GHGNN(gh_gnn_models, T=T, AD=None)
                    system = margules_system(*ln_gammas_inf)
                    
                    def Rela_Vola(SF):
                        '''
                        Objective function to get minSF for rv=rv_threshold
                        
                        It gets the minimum relative volatility accross all 
                        compositions to ensure that the minSF is really the 
                        minimum among all possible compositions.

                        '''
                        x_i, x_j, x_k = get_xs_for_SF(SF)
                        gamma_i, gamma_j, gamma_k = system.get_gammas(x_i, x_j, x_k)
                        relative_volatilities = get_relative_volatility(P_i, P_j, gamma_i, gamma_j)
                        return (rv_threshold - min(relative_volatilities))**2
                        
                    # Minimize function to get minSF
                    results = minimize_scalar(Rela_Vola, bounds=(0,10000), 
                                              method='bounded', options={'maxiter':2000})
                    minSFs[s,t] = results.x
                    
            if AD is not None:
                df_AD = pd.DataFrame({
                    'Solvent_SMILES':solvents,
                    'Solvent_name':solvents_names,
                    'n_class_i': n_class_i_storage,
                    'n_class_j': n_class_j_storage,
                    'max_10_sim_i': max_10_sim_i_storage,
                    'max_10_sim_j': max_10_sim_j_storage,
                    })
                df_AD.to_csv(ad_file, index=False)
                print(' ------ > AD computed!\n')
        
        df_results = pd.DataFrame(minSFs, columns=Ts)
        df_results.insert(0, 'Solvent_SMILES', solvents)
        df_results.insert(1, 'Solvent_name', solvents_names)
        df_results.to_csv('results/'+mixture_type+'/'+mixture+'_minSF.csv', index=False)
