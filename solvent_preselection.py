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
import matplotlib.pyplot as plt
import matplotlib as mpl
from thermo.unifac import UNIFAC
from GHGNN import GH_GNN
import os

_df_antoine = pd.read_csv('data/Antoine_constants.csv')

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

def get_gammas_GHGNN(K1s, K2s, T):
    ln_gs = []
    for K1, K2 in zip(K1s, K2s):
        ln_g = K1 + K2/T
        ln_gs.append(ln_g)
    return ln_gs

def get_constants_GHGNN(gh_gnn_models):
    K1s, K2s = [], []
    for gh_gnn in gh_gnn_models:
        K1, K2 = gh_gnn.predict(T=0, constants=True) # Rememeber T is a dumb T here
        K1s.append(K1)
        K2s.append(K2)
    return K1s, K2s

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
        
    def get_vapor_pressures(self, c_i, c_j, Ts):
        n_Ts = len(Ts)
        # Get vapor pressures for all temperatures
        P_i_lst = np.zeros(n_Ts)
        P_j_lst = np.zeros(n_Ts)
        for t, T in enumerate(Ts):
            # Compute vapor pressures
            Ai, Bi, Ci = get_Antoine_constants(c_i['smiles'], T)
            P_i_lst[t] = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
            
            Aj, Bj, Cj = get_Antoine_constants(c_j['smiles'], T)
            P_j_lst[t] = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
        return P_i_lst, P_j_lst
        
    
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
        
        # Get vapor pressures for all temperatures
        P_i_lst, P_j_lst = self.get_vapor_pressures(c_i, c_j, Ts)
        
        # Check whether AD was computed before, in case not get storage
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
            # Initialize GH-GNN models
            GH_GNN_i = GH_GNN(c_i['smiles'], solvent)
            GH_GNN_j = GH_GNN(c_j['smiles'], solvent)
            
            # Get AD
            if AD == 'both':
                feasible_sys_i, n_class_i_storage[s], max_10_sim_i_storage[s] = GH_GNN_i.get_AD(AD=AD)
                feasible_sys_j, n_class_j_storage[s], max_10_sim_j_storage[s] = GH_GNN_j.get_AD(AD=AD) 
            elif AD is None:
                pass
            else:
                raise Exception('Current implementation only supports AD="both" or AD=None')
                
            # Get K1 and K2 constants, the temperature pass here could be any float or int
            K1_i, K2_i = GH_GNN_i.predict(T=0, constants=True)
            K1_j, K2_j = GH_GNN_j.predict(T=0, constants=True)
        
            for t, T in enumerate(Ts):
                
                P_i = P_i_lst[t]
                P_j = P_j_lst[t]
                
                ln_gamma_inf_i = K1_i + K2_i/T
                ln_gamma_inf_j = K1_j + K2_j/T
            
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
        
        # Get vapor pressures for all temperatures
        P_i_lst, P_j_lst = self.get_vapor_pressures(c_i, c_j, Ts)
        
        # Check whether AD was computed before, in case not get storage
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
            # Initialize GH-GNN models
            c_k = {'smiles': solvent} 
            gh_gnn_models = get_gh_gnn_models(c_i, c_j, c_k)
            
            GH_GNN_i = gh_gnn_models[2]
            GH_GNN_j = gh_gnn_models[4]
            
            # Get AD
            if AD == 'both':
                feasible_sys_i, n_class_i_storage[s], max_10_sim_i_storage[s] = GH_GNN_i.get_AD(AD=AD)
                feasible_sys_j, n_class_j_storage[s], max_10_sim_j_storage[s] = GH_GNN_j.get_AD(AD=AD) 
            elif AD is None:
                pass
            else:
                raise Exception('Current implementation only supports AD="both" or AD=None')
        
            # Get K1 and K2 constants
            K1s, K2s = get_constants_GHGNN(gh_gnn_models)
            
            K1_i = K1s[2]
            K2_i = K2s[2]
            K1_j = K1s[4]
            K2_j = K2s[4]
            
            for t, T in enumerate(Ts):
                
                P_i = P_i_lst[t]
                P_j = P_j_lst[t]
                
                ln_gamma_inf_i = K1_i + K2_i/T
                ln_gamma_inf_j = K1_j + K2_j/T
                
                gamma_inf_i = np.exp(ln_gamma_inf_i)
                gamma_inf_j = np.exp(ln_gamma_inf_j)
                
                a_inf = get_relative_volatility(P_i, P_j, gamma_inf_i, gamma_inf_j)
                
                if a_inf < rv_threshold:
                     minSFs[s,t] = np.nan
                else:
                    ln_gammas_inf = get_gammas_GHGNN(K1s, K2s, T=T)
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



def normalize_entrainer_free(x_i, y_i, x_j, y_j):
    # Normalization to entrainer-free basis
    x_i = x_i/(x_i + x_j)
    y_i = y_i/(y_i + y_j)
    x_j = 1-x_i
    y_j = 1-y_i
    return (x_i, y_i), (x_j, y_j)
    

def get_pseudo_binary_VLE_isothermal(c_i:dict, c_j:dict, T:float, 
                                     margules_system, SF:float):
    '''
    Computes the vapor and liquid fractions of the key components in the 
    pseudo-binary VLE at the given temperature

    Parameters
    ----------
    c_i : dict
        Info of component i
    c_j : dict
        Info of component j
    T : float
        Temperature of the system in K.
    margules_system : class
        Initialized margules system with the corresponding 6 infinite dilution 
        activity coefficient values.
    SF : float
        Solvent to feed ratio.

    Returns
    -------
    tuple
        liquid and vapor fraction of components i and j 
        (x_i, y_i), (x_j, y_j).

    '''
    x_i, x_j, x_k = get_xs_for_SF(SF)
    
    gamma_i, gamma_j, gamma_k = margules_system.get_gammas(x_i, x_j, x_k)
    
    smiles_i,  smiles_j = c_i['smiles'], c_j['smiles']
    
    Ai, Bi, Ci = get_Antoine_constants(smiles_i, T)
    Aj, Bj, Cj = get_Antoine_constants(smiles_j, T)
    
    P_i = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
    P_j = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
    
    p_i = x_i*gamma_i*P_i
    p_j = x_j*gamma_j*P_j
    P   = p_i + p_j
    
    y_i = p_i/P
    y_j = p_j/P
    
    (x_i, y_i), (x_j, y_j) = normalize_entrainer_free(x_i, y_i, x_j, y_j)
    
    relative_volatilities = get_relative_volatility(P_i, P_j, gamma_i, gamma_j)
    
    return [(x_i, y_i), (x_j, y_j)], relative_volatilities

def get_pseudo_binary_VLE_isothermal_thermo(c_i:dict, c_j:dict, c_k:dict, T:float,
                                            SF:float, model:str):
    
    x_i, x_j, x_k = get_xs_for_SF(SF)
    
    Ai, Bi, Ci = get_Antoine_constants(c_i['smiles'], T)
    Aj, Bj, Cj = get_Antoine_constants(c_j['smiles'], T)
    P_i = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
    P_j = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
    
    y_i, y_j = np.zeros(x_i.shape[0]), np.zeros(x_i.shape[0])
    gamma_i = np.zeros(x_i.shape[0])
    gamma_j = np.zeros(x_i.shape[0])
    for i in range(x_i.shape[0]):
        xs = [x_i[i], x_j[i], x_k[i]]
        
        if model == 'UNIFAC_Do':
            GE = UNIFAC.from_subgroups(chemgroups=[c_i['UNIFAC_Do groups'], c_j['UNIFAC_Do groups'], c_k['UNIFAC_Do groups']], 
                                        T=T, 
                                        xs=xs, 
                                        version=1)
        else:
            pass
        
        
        gamma_1, gamma_2, gamma_3 = GE.gammas()
        p_i = x_i[i]*gamma_1*P_i
        p_j = x_j[i]*gamma_2*P_j
        P_calc   = p_i + p_j
        
        y_i[i] = p_i/P_calc
        y_j[i] = p_j/P_calc
        
        gamma_i[i] = gamma_1
        gamma_j[i] = gamma_2
        
    (x_i, y_i), (x_j, y_j) = normalize_entrainer_free(x_i, y_i, x_j, y_j)
    
    relative_volatilities = get_relative_volatility(P_i, P_j, gamma_i, gamma_j)
    
    return [(x_i, y_i), (x_j, y_j)], relative_volatilities

def get_pseudo_binary_VLE_isobaric_thermo(c_i:dict, c_j:dict, c_k:dict, P:float,
                                            SF:float, bounds, model:str):
    
    x_i, x_j, x_k = get_xs_for_SF(SF)
    
    # Optimization
    p_is = np.zeros(x_i.shape[0])
    p_js = np.zeros(x_i.shape[0])
    for i in tqdm(range(x_i.shape[0])):
        xs = [x_i[i], x_j[i], x_k[i]]
        
        def error_in_P(T, P_true=P):
            
            Ai, Bi, Ci = get_Antoine_constants(c_i['smiles'], T)
            Aj, Bj, Cj = get_Antoine_constants(c_j['smiles'], T)
            
            P_i = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
            P_j = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
            
            if model == 'UNIFAC_Do':
                GE = UNIFAC.from_subgroups(chemgroups=[c_i['UNIFAC_Do groups'], c_j['UNIFAC_Do groups'], c_k['UNIFAC_Do groups']], 
                                            T=T, 
                                            xs=xs, 
                                            version=1)
            else:
                pass
            
            
            gamma_i, gamma_j, gamma_k = GE.gammas()
            
            p_i = x_i[i]*gamma_i*P_i
            p_j = x_j[i]*gamma_j*P_j
            P_calc   = p_i + p_j
            
            return np.abs(P_calc - P_true)
        
        results = minimize_scalar(error_in_P, bounds=bounds, method='bounded')
        
        T = results.x
        
        Ai, Bi, Ci = get_Antoine_constants(c_i['smiles'], T)
        Aj, Bj, Cj = get_Antoine_constants(c_j['smiles'], T)
        
        P_i = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
        P_j = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
        
        if model == 'UNIFAC_Do':
            GE = UNIFAC.from_subgroups(chemgroups=[c_i['UNIFAC_Do groups'], c_j['UNIFAC_Do groups'], c_k['UNIFAC_Do groups']], 
                                        T=T, 
                                        xs=xs, 
                                        version=1)
        else:
            pass
        
        gamma_i, gamma_j, gamma_k = GE.gammas()
        
        p_is[i] = x_i[i]*gamma_i*P_i
        p_js[i] = x_j[i]*gamma_j*P_j
    
    y_i = p_is/P
    y_j = p_js/P

    (x_i, y_i), (x_j, y_j) = normalize_entrainer_free(x_i, y_i, x_j, y_j)
    
    return (x_i, y_i), (x_j, y_j)
    

def get_pseudo_binary_VLE_isobaric(c_i:dict, c_j:dict, P:float, 
                                     K1s:list, K2s:list, SF:float, bounds):
    '''
    Computes the vapor and liquid fractions of the key components in the 
    pseudo-binary VLE at the given pressure in bar

    Parameters
    ----------
    c_i : dict
        Info of component i
    c_j : dict
        Info of component j
    P : float
        Pressure of the system in bar.
    K1s : list
        Collection of K1 parameters for GH-GNN models.
    K2s : list
         Collection of K2 parameters for GH-GNN models.
    SF : float
        Solvent to feed ratio.
    bounds : tuple
        Temperature bounds

    Returns
    -------
    tuple
        liquid and vapor fraction of components i and j 
        (x_i, y_i), (x_j, y_j).

    '''
    x_i, x_j, x_k = get_xs_for_SF(SF)
    smiles_i,  smiles_j = c_i['smiles'], c_j['smiles']
    
    # Optimization
    p_is = np.zeros(x_i.shape[0])
    p_js = np.zeros(x_i.shape[0])
    for i in tqdm(range(x_i.shape[0])):
    
        def error_in_P(T, P_true=P):
            
            Ai, Bi, Ci = get_Antoine_constants(smiles_i, T)
            Aj, Bj, Cj = get_Antoine_constants(smiles_j, T)
            
            P_i = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
            P_j = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
            
            ln_gammas = get_gammas_GHGNN(K1s, K2s, T=T)
            system = margules_system(*ln_gammas)
            
            gamma_i, gamma_j, gamma_k = system.get_gammas(x_i[i], x_j[i], x_k[i])
            
            p_i = x_i[i]*gamma_i*P_i
            p_j = x_j[i]*gamma_j*P_j
            P_calc   = p_i + p_j
            
            return np.abs(P_calc - P_true)
        
        results = minimize_scalar(error_in_P, bounds=bounds, method='bounded', options={'maxiter':2000})
        
        T = results.x
        
        Ai, Bi, Ci = get_Antoine_constants(smiles_i, T)
        Aj, Bj, Cj = get_Antoine_constants(smiles_j, T)
        
        P_i = get_vapor_pressure_Antoine(Ai, Bi, Ci, T)
        P_j = get_vapor_pressure_Antoine(Aj, Bj, Cj, T)
        
        ln_gammas = get_gammas_GHGNN(K1s, K2s, T=T)
        system = margules_system(*ln_gammas)
        
        gamma_i, gamma_j, gamma_k = system.get_gammas(x_i[i], x_j[i], x_k[i])
        
        p_is[i] = x_i[i]*gamma_i*P_i
        p_js[i] = x_j[i]*gamma_j*P_j
        
    y_i = p_is/P
    y_j = p_js/P
    
    (x_i, y_i), (x_j, y_j) = normalize_entrainer_free(x_i, y_i, x_j, y_j)
    
    return (x_i, y_i), (x_j, y_j)


        

def plot_pseudoVLE(SFs, VLEs_lst, exp_lst, c_1, c_2, c_3, T_P, 
                              folder_figures, mode, model):
    # Set the default color cycle
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#b3cde0', '#6497b1', '#005b96', '#03396c', '#011f4b']) 
    mks = ['*', '^', 's', 'd', 'x']
    
    fig = plt.figure(figsize=(6,5))
    plt.plot([0,1], [0,1], 'k--', label='x=y')
    for i, SF in enumerate(SFs):
        (x_i, y_i), (x_j, y_j) = VLEs_lst[i]
        plt.plot(x_i, y_i, '-', label='SF ' + str(SF))
        
    for i, d in enumerate(exp_lst):
        sf_exp = d['SF']
        df_exp = d['df']
        plt.plot(df_exp['x'], df_exp['y'], 'k'+mks[i], label='Exp. SF='+str(sf_exp))
    ax = plt.gca()
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Molar liquid fraction ' + c_1['name'], fontsize=14)
    plt.ylabel('Molar vapor fraction ' + c_1['name'], fontsize=14)
    plt.legend(ncol=2)
    if mode == 'T':
        plt.title(f'Isothermal pseudo-binary VLE at {T_P} K ', fontsize=14)
        fig_name = '/'+model+'_isothermal_'+str(T_P)
    elif mode == 'P':
        plt.title(f'Isobaric pseudo-binary VLE at {T_P} bar ', fontsize=14)
        fig_name = '/'+model+'_isobaric_'+str(T_P)
    plt.close(fig)
    name = c_1['name'] + ',' + c_2['name'] + ',' + c_3['name']
    fig.savefig(folder_figures+fig_name+'_'+name+'.png', dpi=300, format='png')
    # fig.savefig(folder_figures+fig_name+'_'+name+'.svg', dpi=300, format='svg')
    
def plot_relative_vola_SFs(SFs, alphas_lst, VLEs_lst, c_1, c_2, c_3, T, 
                              folder_figures, model):
    # Set the default color cycle
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#b3cde0', '#6497b1', '#005b96', '#03396c', '#011f4b']) 
    fig = plt.figure(figsize=(6,5))
    for i, SF in enumerate(SFs):
        (x_i, y_i), (x_j, y_j) = VLEs_lst[i]
        alphas = alphas_lst[i]
        plt.plot(x_i, alphas, '-', label='SF ' + str(SF))
    ax = plt.gca()
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    plt.xlim(0,1)
    plt.xlabel('Molar liquid fraction ' + c_1['name'], fontsize=14)
    plt.ylabel('Pseudo-relative volatility ', fontsize=14)
    plt.legend(ncol=2)
    plt.title(f'Isothermal pseudo-relative volatility at {T} K ', fontsize=14)
    fig_name = '/'+model+'_isothermal_RV_'+str(T)
    
    plt.close(fig)
    name = c_1['name'] + ',' + c_2['name'] + ',' + c_3['name']
    fig.savefig(folder_figures+fig_name+'_'+name+'.png', dpi=300, format='png')
    # fig.savefig(folder_figures+fig_name+'_'+name+'.svg', dpi=300, format='svg')
    
            
   