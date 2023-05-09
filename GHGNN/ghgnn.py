'''
Project: GNN_IAC_T

                    User-friendly GH-GNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

from GHGNN.GHGNN_architecture import GHGNN_model
import torch
from rdkit import Chem
import numpy as np
from rdkit.Chem import rdMolDescriptors
from mordred.Polarizability import APol, BPol
from mordred.TopoPSA import TopoPSA
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import Dataset
import urllib.request 
import json
import time
import pickle
from rdkit.DataStructs import FingerprintSimilarity as FPS
import pathlib

current_path = str(pathlib.Path(__file__).parent.resolve())
##############################
# --- Mol to graph utils --- #
##############################

possible_atom_list = ['C','N','O','Cl','S','F','Br','I','Si','Sn','Pb','Ge',
                      'H','P','Hg', 'Te']

possible_hybridization = [Chem.rdchem.HybridizationType.S,
                          Chem.rdchem.HybridizationType.SP, 
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3]

possible_chiralities =[Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]

possible_num_bonds = [0,1,2,3,4]

possible_formal_charge = [0,1,-1]

possible_num_Hs  = [0,1,2,3]

possible_stereo  = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE]

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    '''
    Get atom features
    '''
    Symbol       = atom.GetSymbol()
    
    # Features
    Type_atom     = one_of_k_encoding(Symbol, possible_atom_list)
    Ring_atom     = [atom.IsInRing()]
    Aromaticity   = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Bonds_atom    = one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    Formal_charge = one_of_k_encoding(atom.GetFormalCharge(), possible_formal_charge)
    num_Hs        = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)
    Type_chirality= one_of_k_encoding(atom.GetChiralTag(), possible_chiralities)
    
    # Merge features in a list
    results = Type_atom + Ring_atom + Aromaticity + Hybridization + \
        Bonds_atom + Formal_charge + num_Hs + Type_chirality
    
    return np.array(results).astype(np.float32)

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def bond_features(bond):
    '''
    Get bond features
    '''
    bt = bond.GetBondType()
    
    type_stereo = one_of_k_encoding(bond.GetStereo(), possible_stereo)
    
    # Features
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()] + type_stereo
    return np.array(bond_feats).astype(np.float32)

###########################################################
# --- Data loader 2Graphs_1Output and system property --- #
###########################################################

class PairDataset_T(Dataset):
    def __init__(self, datasetA, datasetB, datasetT):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetT = datasetT

    def __getitem__(self, idx):
        return self.datasetA[idx], self.datasetB[idx], self.datasetT[idx]
    
    def __len__(self):
        return len(self.datasetA)
    

def collate_T(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, batchC


#################
# --- Model --- #
#################

class GH_GNN():
    def __init__(self, solute_smiles, solvent_smiles):
        
        # SMILES to Molecule
        self.solute = Chem.MolFromSmiles(solute_smiles)
        self.solvent = Chem.MolFromSmiles(solvent_smiles)
        
        if self.solute is None:
            print(f'Current solute smiles ({solute_smiles}) was invalid for rdkit')
            return None
        elif self.solvent is None:
            print(f'Current solvent smiles ({solvent_smiles}) was invalid for rdkit')
            return None
        
        # Molecules to graphs
        self.g_solute, self.g_solvent = self.sys2graphs(self.solute, self.solvent)
        
        self.architecture = GHGNN_model
       
        v_in = 37
        e_in = 9
        u_in = 3 
        hidden_dim = 113
        model    = GHGNN_model(v_in, e_in, u_in, hidden_dim)
        
        path_parameters = current_path+'/GHGNN.pth'
        available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model.load_state_dict(torch.load(path_parameters, 
                                         map_location=torch.device(available_device)))
        self.device   = torch.device(available_device)
        self.model    = model.to(self.device)
        self.first_query = time.time()
        
        self.interpolation = self.check_interpolation(self.solvent, self.solute)
        
    def classify_mol(self, mol):
        inchikey = Chem.inchi.MolToInchiKey(mol)
        url = 'http://classyfire.wishartlab.com/entities/' + str(inchikey) + '.json'
        
        try:
            with urllib.request.urlopen(url) as webpage:
                data = json.loads(webpage.read().decode())    
            if data['class']['name'] is None:
                raise Exception()
            return data
        except:
            return None
    
    def check_interpolation(self, solvent, solute):
        
        solvent_smiles = Chem.MolToSmiles(solvent)
        solute_smiles = Chem.MolToSmiles(solute)
        
        with open(current_path+'/training_solutes.pickle', 'rb') as handle:
            training_solutes = pickle.load(handle)
            
        with open(current_path+'/training_solvents.pickle', 'rb') as handle:
            training_solvents = pickle.load(handle)
        
        self.solvent_in_training = solvent_smiles in training_solvents
        self.solute_in_training = solute_smiles in training_solutes
            
        if self.solvent_in_training and self.solute_in_training:
            return True
        else:
            return False
        
    
    def indicator_class(self, solvent, solute):
        solute_class = self.classify_mol(solute)
        solvent_class = self.classify_mol(solvent)
        time.sleep(1)
        
        solute_class = solute_class['class']['name'] if solute_class != None else ''
        solvent_class = solvent_class['class']['name'] if solvent_class != None else ''
        
        key_class = solvent_class + '_' + solute_class
        
        with open(current_path+'/training_classes.pickle', 'rb') as handle:
            training_classes = pickle.load(handle)
            
        try:
            n_observations = training_classes[key_class]
        except:
            n_observations = 0
            pass
        return n_observations
    
    def indicator_tanimoto(self, solvent, solute):
        solvent_fp = Chem.RDKFingerprint(solvent)
        solute_fp = Chem.RDKFingerprint(solute)
        
        with open(current_path+'/training_fp.pickle', 'rb') as handle:
            fps_training = pickle.load(handle)
        
        if not self.solvent_in_training and not self.solute_in_training:
            similarities_solv = sorted([FPS(solvent_fp, fp_train) for fp_train in fps_training])
            similarities_solu = sorted([FPS(solute_fp, fp_train) for fp_train in fps_training])
            max_10_sim_solv = np.mean(similarities_solv[-10])
            max_10_sim_solu = np.mean(similarities_solu[-10])
            max_10_sim = min(max_10_sim_solv, max_10_sim_solu)
        elif self.solvent_in_training:
            similarities_solu = sorted([FPS(solute_fp, fp_train) for fp_train in fps_training])
            max_10_sim = np.mean(similarities_solu[-10])
        elif self.solute_in_training:
            similarities_solv = sorted([FPS(solvent_fp, fp_train) for fp_train in fps_training])
            max_10_sim = np.mean(similarities_solv[-10])
        return max_10_sim
    
    def sys2graphs(self, solute, solvent):
        atoms_solu  = solute.GetAtoms()
        bonds_solu  = solute.GetBonds()
        
        atoms_solv  = solvent.GetAtoms()
        bonds_solv  = solvent.GetBonds()
        
        # Information on nodes
        node_f_solv = [atom_features(atom) for atom in atoms_solv]
        node_f_solu = [atom_features(atom) for atom in atoms_solu]
        
        # Information on edges
        edge_index_solv = get_bond_pair(solvent)
        edge_attr_solv  = []
        
        for bond in bonds_solv:
            edge_attr_solv.append(bond_features(bond))
            edge_attr_solv.append(bond_features(bond))
            
        edge_index_solu = get_bond_pair(solute)
        edge_attr_solu  = []
        
        for bond in bonds_solu:
            edge_attr_solu.append(bond_features(bond))
            edge_attr_solu.append(bond_features(bond))
            
            
        # Atomic polarizability
        calc = APol()
        ap_solv = calc(solvent)
        ap_solu = calc(solute)
        
        # Bond polarizability
        calc = BPol()
        bp_solv = calc(solvent)
        bp_solu = calc(solute)
        
        # Topological Polar Surface Area
        calc = TopoPSA()
        topopsa_solv = calc(solvent)
        topopsa_solu = calc(solute)
        
        # Intra hydrogen-bond acidity and basicity
        hb_solv = min(rdMolDescriptors.CalcNumHBA(solvent), rdMolDescriptors.CalcNumHBD(solvent))
        hb_solu = min(rdMolDescriptors.CalcNumHBA(solute), rdMolDescriptors.CalcNumHBD(solute))
        
        # Inter hydrogen-bond
        inter_hb = min(rdMolDescriptors.CalcNumHBA(solvent), rdMolDescriptors.CalcNumHBD(solute)) \
                             + min(rdMolDescriptors.CalcNumHBA(solute),rdMolDescriptors.CalcNumHBD(solvent))
                             
        # Store all information in a graph
        nodes_info_solv = torch.tensor(np.array(node_f_solv), dtype=torch.float)
        edges_indx_solv = torch.tensor(np.array(edge_index_solv), dtype=torch.long)
        edges_info_solv = torch.tensor(np.array(edge_attr_solv), dtype=torch.float)
        graph_solv = Data(x=nodes_info_solv, edge_index=edges_indx_solv, edge_attr=edges_info_solv,
                          ap=ap_solv,
                          bp=bp_solv,
                          topopsa=topopsa_solv,
                          hb=hb_solv, 
                          inter_hb=inter_hb,
                          y=torch.tensor([1]))
        
        nodes_info_solu = torch.tensor(np.array(node_f_solu), dtype=torch.float)
        edges_indx_solu = torch.tensor(np.array(edge_index_solu), dtype=torch.long)
        edges_info_solu = torch.tensor(np.array(edge_attr_solu), dtype=torch.float)
        graph_solu = Data(x=nodes_info_solu, edge_index=edges_indx_solu, edge_attr=edges_info_solu,
                          ap=ap_solu,
                          bp=bp_solu,
                          topopsa=topopsa_solu, 
                          hb=hb_solu, 
                          inter_hb=inter_hb,
                          y=torch.tensor([1]))
        
        return graph_solu, graph_solv
    
    def get_AD(self, AD='both'):
        solute = self.solute
        solvent = self.solvent
        
        # Applicability domain
        feasible_sys = True
        if AD=='both':
            if self.interpolation:
                feasible_sys = True
                max_10_sim = 9999 # Just a large number for indicating interpolation
                n_class = 9999 # Just a large number for indicating interpolation
            else:
                # Tanimoto similarity indicator
                max_10_sim = self.indicator_tanimoto(solvent, solute)
                if max_10_sim < 0.35:
                    feasible_sys = False
                
                # Chemical class indicator
                n_class = self.indicator_class(solvent, solute)
                if n_class < 25:
                    feasible_sys = False
        elif AD=='class':
            if self.interpolation:
                feasible_sys = True
                n_class = 9999 # Just a large number for indicating interpolation
            else:
                # Chemical class indicator
                n_class = self.indicator_class(solvent, solute)
                if n_class < 25:
                    feasible_sys = False
        elif AD=='tanimoto':
            if self.interpolation:
                feasible_sys = True
                max_10_sim = 9999 # Just a large number for indicating interpolation
            else:
                # Tanimoto similarity indicator
                max_10_sim = self.indicator_tanimoto(solvent, solute)
                if max_10_sim < 0.35:
                    feasible_sys = False
        elif AD is None:
            feasible_sys = None
        else:
            raise Exception('Invalid value for AD')
        if AD is None:
            return None
        elif AD == 'class':
            return feasible_sys, n_class
        elif AD == 'tanimoto':
            return feasible_sys, max_10_sim
        elif AD == 'both':
            return feasible_sys, n_class, max_10_sim
    
    def predict(self, T, constants=False):
        
        T = T - 273.15 # GHGNN is trained with T in C
        
        g_solute, g_solvent = self.g_solute, self.g_solvent
        
        g_T = Data(x=torch.tensor(T, dtype=torch.float).reshape(1))
        
        pair_dataset = PairDataset_T([g_solvent], [g_solute], [g_T])
        
        predict_loader = torch.utils.data.DataLoader(pair_dataset, 
                                                     batch_size=1, 
                                                     shuffle=False, 
                                                     drop_last=False, 
                                                     collate_fn=collate_T)
        
        with torch.no_grad():
            for batch_solvent, batch_solute, batch_T in predict_loader:
                batch_solvent = batch_solvent.to(self.device)
                batch_solute  = batch_solute.to(self.device)
                batch_T       = batch_T.to(self.device)
               
                if torch.cuda.is_available():
                    pred  = self.model(batch_solvent.cuda(), batch_solute.cuda(), batch_T.cuda(), constants=constants).cpu()
                else:
                    pred  = self.model(batch_solvent, batch_solute, batch_T, constants=constants)
                if constants:
                    K1, K2 = pred
                    K1 = K1.numpy().reshape(-1,)[0]
                    K2 = K2.numpy().reshape(-1,)[0]
                else:
                    ln_gamma_ij  = pred.numpy().reshape(-1,)[0]
        if constants:
            return K1, K2
        else:
            return ln_gamma_ij
        