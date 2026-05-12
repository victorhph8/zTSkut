from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import LeakyReLU # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator # type: ignore
from mendeleev import element
from pymatgen.core.periodic_table import Element
import periodictable
import os
from joblib import load
from tqdm import tqdm
from icecream import ic

ic.disable()


# ---------- CONFIG ----------
# Descriptor-level CSV for unknown systems
# Saved scaler + model from training
SCALER_FILE        = "scaler_ZT.joblib"
FEATURES_FILE      = "feature_columns_ZT.txt"
MODEL_FILE         = "model_keras_skutt.h5"

# Output files
# ----------------------------


#############################################
#### Descriptors using RDKit & Mendeleev ####
#############################################

# Definitions to identify all possible elements in compositions
def alphabets(element):
    return ''.join(filter(str.isalpha, element))

def unique_elements(final_extract, list_comp):
    prelim_unique = []
    for column in list_comp:
        unique_column = final_extract[column].unique()
        unique_column = unique_column.astype(str) # This line is different from original 'Keras' script, due to new data (to be predicted) conversion to string type was needed
        prelim_unique.append(unique_column)
    prelim_unique_df = pd.DataFrame(np.concatenate(prelim_unique), columns=['prelim'])
    prelim_unique_no_num = prelim_unique_df.loc[:,'alphabets'] = [alphabets(x) for x in prelim_unique_df['prelim']]
    prelim_unique_no_num_df = pd.DataFrame(prelim_unique_no_num, columns=['prelim_nonum'])
    no_empty = prelim_unique_no_num_df['prelim_nonum'] != '' # Detect empty with booleans
    prelim_unique_no_num_no_empty_df = prelim_unique_no_num_df[no_empty]
    all_unique = prelim_unique_no_num_no_empty_df['prelim_nonum'].unique()
    return(all_unique)

# Definition to load descriptors using RDKit, Mendeleev (incomplete for some elements) & Pymatgen
def load_descriptors(atom):
    #if type(atom) == int or type(atom) == float or atom == '0':
    #if atom == 'Al' or atom == 'Mg' or atom == 'Li' or atom == 'Sn' or atom == 'Se' or atom == 'Br' or atom == 'S' or atom == 'Gd' or atom == 'Tb' or atom == 'Dy' or atom == 'In' or atom == 'K' or atom == 'Sm' or atom == 'Yb' or atom == 'Eu' or atom == 'Nd' or atom == 'Pr' or atom == 'Ni' or atom == 'Pd' or atom == 'Te' or atom == 'Ga' or atom == 'Ge' or atom == 'Cr' or atom == 'Ce' or atom == 'Fe' or atom == 'Co' or atom == 'Sb' or atom == 'Ca' or atom == 'Sr' or atom == 'Ba' or atom == 'La':
    elements = [element.symbol for element in periodictable.elements]
    if atom in elements:
        # RDKit (Valence electrons)
        smiles_string = f'[{atom}]'
        mol = Chem.MolFromSmiles(smiles_string) # type: ignore
        chosen_descriptors = ['NumValenceElectrons']
        mol_descriptor_calculator = MolecularDescriptorCalculator(chosen_descriptors)
        val_e = list(mol_descriptor_calculator.CalcDescriptors(mol))
        # Mendeleev (Ionisation Pot, Atomic Mass)
        ip = element(atom).ionenergies
        #ea = element(atom).electron_affinity
        mass = element(atom).mass
        #elecneg_pauling = element(atom).en_pauling
        # Pymatgen (Electron Aff, Electroneg)
        ea = Element(atom).electron_affinity
        elecneg_pauling = Element(atom).X
        # Summary of our new descriptors (RDKit + Mendeleev: [val_e, ip, ea, mass, elecneg])
        list_descriptors = [val_e[0], ip.get(1), ea, mass, elecneg_pauling] # type:ignore
        ic(list_descriptors)
        return(list_descriptors)
        #return(','.join(str(x) for x in list_descriptors)) # Returns descriptors as str instead of list (great for training our model!)
    else:
        return([0,0,0,0,0])

# Detect number of anions
def num_anions(atom):
    num_anions_all = []
    elements = [element.symbol for element in periodictable.elements]
    num_anions = 0
    if atom in elements:
        num_anions +=1
        num_anions_all.append(num_anions)
        return(','.join(str(x) for x in num_anions_all))
    else:
        return(atom)

# Detect number of cations
def num_cations(atom):
    num_cations_all = []
    elements = [element.symbol for element in periodictable.elements]
    num_cations = 0
    if atom in elements:
        num_cations +=1
        num_cations_all.append(num_cations)
        return(','.join(str(x) for x in num_cations_all))
    else:
        return(atom)

# Detect number of fillers    
def num_fillers(atom):
    num_fillers_all = []
    elements = [element.symbol for element in periodictable.elements]
    num_fillers = 0
    if atom in elements:
        num_fillers +=1
        num_fillers_all.append(num_fillers)
        return(','.join(str(x) for x in num_fillers_all))
    else:
        return(atom)

def load_feature_order(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]

def main(input_csv="samples_file.csv"):
    # Definition to copy descriptors extracted from database of unique elements - Moved from original place in 'Keras-NN_V6_Mallos.py'
    def load_unique_descriptors(atom):
        elements = [element.symbol for element in periodictable.elements]
        #dict_descriptors = unique_atoms_descriptors_df.to_dict('records')
        if atom in elements:
            descriptors_final = unique_atoms_descriptors_df.loc[unique_atoms_descriptors_df['elements'] == atom, 'descriptors']
            flattened_list_descriptors_final = [x for sublist in descriptors_final for x in sublist] #type:ignore
            ic(flattened_list_descriptors_final)
            return(flattened_list_descriptors_final)
        else:
            return([0,0,0,0,0])

    # Load new data for prediction
    target = 'ZT'
    read_samples = pd.read_csv(input_csv)
    if target not in read_samples.columns:
        read_samples[target] = 0.0
    cols = read_samples.columns
    ic(list(cols))
    read_samples_np = read_samples.to_numpy()
    ic(read_samples_np)
    raw_label = target
    
    # Always generate descriptors from the uploaded CSV and scale using the saved training scaler.
    if True:
        print("* Generating descriptors and scaling with the saved training scaler...")

        #######################################################
        #### Replace atom names with descriptors - PHASE 1 ####
        #######################################################
        final_extract = read_samples
        final_extract_descriptors = final_extract.copy()
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        ic(final_extract_descriptors)
        ic(final_extract_descriptors['n_300'].dtypes)

        # Extract number of anions, cations and fillers of each system (row)
        anions = ['a_comp_1','a_comp_2','a_comp_3']
        cations = ['c_comp_1','c_comp_2','c_comp_3']
        fillers = ['f_comp_1','f_comp_2','f_comp_3','f_comp_4','f_comp_5']
        # Number of anions of each system
        num_anions_all_post = []
        for column in anions:
            #ic(column)
            #ic(final_extract_descriptors[column])
            num_anions_post = final_extract[column].apply(num_anions)
            #ic(num_anions_post)
            num_anions_all_post.append(list(num_anions_post)) # Collect info from definition in each row
        num_anions_all_post = np.array(num_anions_all_post, dtype=float) # Transform it to NumPy array to be considered as a float (int or str won't work)
        sum_anions = np.array(sum(num_anions_all_post), dtype=float) # Sum numbers of each row to have total num of anions
        df_sum_anions = pd.DataFrame(sum_anions, columns=['num_anions']) # Transform last NumPy array to dataframe so we can concatenate in a single df
        ic(df_sum_anions)
            #ic(final_extract[column].apply(num_anions))
            #final_extract_descriptors[column] = final_extract[column].apply(num_anions)
        # Number of cations of each system
        num_cations_all_post = []
        for column in cations:
            #ic(column)
            #ic(final_extract_descriptors[column])
            num_cations_post = final_extract[column].apply(num_cations)
            #ic(num_cations_post)
            num_cations_all_post.append(list(num_cations_post)) # Collect info from definition in each row
        num_cations_all_post = np.array(num_cations_all_post, dtype=float) # Transform it to NumPy array to be considered as a float (int or str won't work)
        sum_cations = np.array(sum(num_cations_all_post), dtype=float) # Sum numbers of each row to have total num of cations
        df_sum_cations = pd.DataFrame(sum_cations, columns=['num_cations']) # Transform last NumPy array to dataframe so we can concatenate in a single df
        ic(df_sum_cations)
        # Number of fillers of each system
        num_fillers_all_post = []
        for column in fillers:
            #ic(column)
            #ic(final_extract_descriptors[column])
            num_fillers_post = final_extract[column].apply(num_fillers)
            #ic(num_fillers_post)
            num_fillers_all_post.append(list(num_fillers_post)) # Collect info from definition in each row
        num_fillers_all_post = np.array(num_fillers_all_post, dtype=float) # Transform it to NumPy array to be considered as a float (int or str won't work)
        sum_fillers = np.array(sum(num_fillers_all_post), dtype=float) # Sum numbers of each row to have total num of fillers
        df_sum_fillers = pd.DataFrame(sum_fillers, columns=['num_fillers']) # Transform last NumPy array to dataframe so we can concatenate in a single df
        ic(df_sum_fillers)

        num_cat_an_fill = pd.concat([df_sum_anions, df_sum_cations, df_sum_fillers], axis=1) # Summary (dataframe) of numbers of all anions, cations and fillers of each row (concatenated in final csv - Phase 1)
        ic(num_cat_an_fill)

        # Replace all atoms with descriptors
        print('* Inspecting and replacing all atoms with their descriptors...')
        list_comp = ['a_comp_1', 'a_comp_2', 'a_comp_3', 'c_comp_1', 'c_comp_2', 'c_comp_3', 'f_comp_1', 'f_comp_2', 'f_comp_3', 'f_comp_4', 'f_comp_5']
        #for column in list(all_expanded_queries_append.columns): # Too general :( 'list_comp' should focus only on compositions and nothing else ;)
        unique_atoms = unique_elements(final_extract_descriptors, list_comp) # Use 'unique_elements' def to detect all atoms in your database
        print(f'   ++ Elements in your database: {list(unique_atoms)}')
        unique_atoms_df = pd.DataFrame(unique_atoms, columns=['elements'])
        unique_descriptors_series = unique_atoms_df['elements'].apply(load_descriptors) # Creating series in Pandas to have unique descriptors (useful to avoid calling RDKit, Mendeleev and Pymatgen housands of times! This should save A LOT of time!!!)
        unique_descriptors_series.name = 'descriptors' # Changing name of series (different commands to change columns' names in a Pandas dataframe) 
        unique_atoms_descriptors_df = pd.concat([unique_atoms_df, unique_descriptors_series], axis=1) # Creating a unique dataframe with all descriptors (like a dictionary)
        ic(unique_atoms_descriptors_df)
        unique_atoms_descriptors_df.to_csv('unique_descriptors_elements.csv', index=False)

        # Use def 'load_unique_descriptors' to only copy descriptors for each element
        for column in tqdm(list_comp): # type: ignore
            ic(column)
            final_extract_descriptors[column] = final_extract[column].apply(load_unique_descriptors)
        ic(final_extract_descriptors)
        ic(final_extract_descriptors['c_comp_2'])
        ic(final_extract_descriptors['c_frac_2'])
        # Detect nested descriptors
        # Use the first row instead of iloc[5], because web-app users may submit only one system.
        if len(final_extract_descriptors) == 0:
            raise ValueError("No descriptors were generated from the input CSV.")
        
        sizes_descriptors = final_extract_descriptors.iloc[0].map(lambda x: np.size(x))
        ic(sizes_descriptors)
        
        size_descriptors_df = pd.DataFrame(
            sizes_descriptors.tolist()
        ).pivot_table(columns=list(final_extract_descriptors.columns))
        
        ic(size_descriptors_df)
        
        all_nested_descriptors = []
        for i in size_descriptors_df.columns:
            check_sizes_descriptors = size_descriptors_df.loc[lambda df: df[i] > 1] # Creates new dataframe including only the nested query
            if (not check_sizes_descriptors.empty):
                nested_descriptors = i # Identifies every single nested array in size_query_df
                all_nested_descriptors.append(nested_descriptors) # Unique queries that have nested arrays
        ic(all_nested_descriptors) 
        size_all_nested_descriptors = sizes_descriptors.loc[all_nested_descriptors].tolist()
        ic(size_all_nested_descriptors)

        all_summary_nested_list_descriptors = []
        summary_nested_tuple_descriptors = list(zip(all_nested_descriptors, size_all_nested_descriptors)) # Join both type and size - output=tuple :( - should be list
        for i in summary_nested_tuple_descriptors:
            summary_nested_list_descriptors = list(i)
            all_summary_nested_list_descriptors.append(summary_nested_list_descriptors) # Converted to list each of them and ready to work
        ic(all_summary_nested_list_descriptors)

        # Descriptors - Loop used for 'FOR' loop with 2 simultaneous indices (1.- Type of nested query and 2.- Size of nested query)
        all_expanded_descriptors = []
        names_descriptors = ['val_e', 'ip', 'ea', 'mass', 'elecneg']
        for k in all_summary_nested_list_descriptors:
            ic(k[0], k[1]) # simultaneous indices
            nested_arrays_descriptors = pd.DataFrame(final_extract_descriptors[f'{k[0]}']) # 1st index (type of query)
            columns_descriptors = []
            for column_descriptors in range(k[1]): # 2nd index (size)
                columns_descriptors.append(f'{k[0]}_{names_descriptors[column_descriptors]}') # Columns' name for expanded nested queries
            ic(columns_descriptors)
            single_col_nested_arrays_descriptors = nested_arrays_descriptors.explode(f'{k[0]}') # This will list as a single column all the nested query
            expanded_query_descriptors = pd.DataFrame(single_col_nested_arrays_descriptors.values.reshape(len(nested_arrays_descriptors),k[1]), columns=columns_descriptors) # Recover shape of original query in different columns
            ic(expanded_query_descriptors)
            all_expanded_descriptors.append(expanded_query_descriptors)

        # Descriptors - All expanded queries concatenated in a single dataframe
        all_expanded_descriptors_append = pd.concat(all_expanded_descriptors, axis=1)
        ic(all_expanded_descriptors_append)
        # Descriptors - Add expanded dataframes (all_expanded_queries_append) to original dataframe (extract)
        final_extract_descriptors_2 = pd.concat([num_cat_an_fill, all_expanded_descriptors_append, final_extract_descriptors], axis=1)
        ic(final_extract_descriptors_2)
        # Descriptors - Delete all original nested queries
        for descriptor in all_nested_descriptors:
            final_extract_descriptors_2.pop(f'{descriptor}')
            ic(final_extract_descriptors_2)
        ic(final_extract_descriptors_2)
        # Update list of descriptors (nested queries removed, actually it's a copy of columns' names of final_extract dataframe ;) )
        updated_queries_descriptors = list(final_extract_descriptors_2.columns)
        ic(updated_queries_descriptors)
        # Checkpoint!! - Create csv file with raw descriptors
        final_extract_descriptors_2.to_csv(f'raw_descriptors-Phase_1_{raw_label}.csv', index=False)

        ###################################################################
        #### Feature Engineering - Descriptors from csv file - PHASE 2 ####
        ###################################################################
        print(f'   ++ Calculating new descriptors per each composition...')
        df = pd.read_csv(f'raw_descriptors-Phase_1_{raw_label}.csv')

        #Columns
        a_elecneg_cols = ['a_comp_1_elecneg', 'a_comp_2_elecneg', 'a_comp_3_elecneg']
        a_ip_cols = ['a_comp_1_ip', 'a_comp_2_ip', 'a_comp_3_ip']
        a_ea_cols = ['a_comp_1_ea', 'a_comp_2_ea', 'a_comp_3_ea']
        f_elecneg_cols = ['f_comp_1_elecneg', 'f_comp_2_elecneg', 'f_comp_3_elecneg', 'f_comp_4_elecneg', 'f_comp_5_elecneg']
        f_ip_cols = ['f_comp_1_ip', 'f_comp_2_ip', 'f_comp_3_ip', 'f_comp_4_ip', 'f_comp_5_ip']
        f_ea_cols = ['f_comp_1_ea', 'f_comp_2_ea', 'f_comp_3_ea', 'f_comp_4_ea', 'f_comp_5_ea']
        c_elecneg_cols = ['c_comp_1_elecneg', 'c_comp_2_elecneg', 'c_comp_3_elecneg']
        c_ip_cols = ['c_comp_1_ip', 'c_comp_2_ip', 'c_comp_3_ip']
        c_ea_cols = ['c_comp_1_ea', 'c_comp_2_ea', 'c_comp_3_ea']
        val_e_cols = ['a_comp_1_val_e', 'a_comp_2_val_e', 'a_comp_3_val_e', 'c_comp_1_val_e', 'c_comp_2_val_e', 'c_comp_3_val_e', 'f_comp_1_val_e', 'f_comp_2_val_e', 'f_comp_3_val_e', 'f_comp_4_val_e', 'f_comp_5_val_e']
        mass_cols = ['a_comp_1_mass', 'a_comp_2_mass', 'a_comp_3_mass', 'c_comp_1_mass', 'c_comp_2_mass', 'c_comp_3_mass', 'f_comp_1_mass', 'f_comp_2_mass', 'f_comp_3_mass', 'f_comp_4_mass', 'f_comp_5_mass']
        num_atoms_cols = ['num_anions', 'num_cations','num_fillers']
        fillers_mass_cols = ['f_comp_1_mass', 'f_comp_2_mass', 'f_comp_3_mass', 'f_comp_4_mass', 'f_comp_5_mass']
        f_frac_cols = ['f_frac_1', 'f_frac_2', 'f_frac_3', 'f_frac_4', 'f_frac_5']
        c_frac_cols = ['c_frac_1', 'c_frac_2', 'c_frac_3']
        a_frac_cols = ['a_frac_1', 'a_frac_2', 'a_frac_3']
        frac_cols = ['a_frac_1', 'a_frac_2', 'a_frac_3', 'c_frac_1', 'c_frac_2', 'c_frac_3', 'f_frac_1', 'f_frac_2', 'f_frac_3', 'f_frac_4', 'f_frac_5']
        t_col = ['T']
        target_col = raw_label # Changed to a more generic name (PF is just one of posible labels to be analised...)
        #Obtain values with Numpy
        a_elecneg = df[a_elecneg_cols].to_numpy()
        a_ip = df[a_ip_cols].to_numpy()
        a_ea = df[a_ea_cols].to_numpy()
        c_elecneg = df[c_elecneg_cols].to_numpy()
        c_ip = df[c_ip_cols].to_numpy()
        c_ea = df[c_ea_cols].to_numpy()
        f_elecneg = df[f_elecneg_cols].to_numpy()
        f_ip = df[f_ip_cols].to_numpy()
        f_ea = df[f_ea_cols].to_numpy()
        val_e = df[val_e_cols].to_numpy()
        mass = df[mass_cols].to_numpy()
        fillers_mass = df[fillers_mass_cols].to_numpy()
        f_frac = df[f_frac_cols].to_numpy()
        c_frac  = df[c_frac_cols].to_numpy()
        a_frac = df[a_frac_cols].to_numpy()
        frac = df[frac_cols].to_numpy()
        num_atoms = df[num_atoms_cols].to_numpy()
        T = df[t_col].to_numpy()
        target = df[target_col].to_numpy() # Changed to a more generic name (PF is just one of posible labels to be analised...)

        #Calculate weighted average electronegativity of each system (anions, cations and fillers)
        num_aver_a_elecneg = np.sum(a_elecneg * a_frac, axis=1) #Calculate numerator and denominator. Axis=1 to iterate over each row.
        den_sum_a_frac = np.sum(a_frac, axis=1)
        aver_a_elecneg = np.divide(num_aver_a_elecneg, den_sum_a_frac) #out=np.zeros_like(num_aver_elecneg), where=den_sum_frac != 0) to use when numerator and denominator could be zero.
        aver_a_elecneg_df = pd.DataFrame(aver_a_elecneg, columns=['aver_a_elecneg']) #Create DataFrame for aver_a_elecneg
        #ic(aver_a_elecneg_df)

        num_aver_c_elecneg = np.sum(c_elecneg * c_frac, axis=1) 
        den_sum_c_frac = np.sum(c_frac, axis=1)
        aver_c_elecneg = np.divide(num_aver_c_elecneg, den_sum_c_frac) #out=np.zeros_like(num_aver_elecneg), where=den_sum_frac != 0) to use when numerator and denominator could be zero.
        aver_c_elecneg_df = pd.DataFrame(aver_c_elecneg, columns=['aver_c_elecneg'])
        #ic(aver_c_elecneg_df)

        num_aver_f_elecneg = np.sum(f_elecneg * f_frac, axis=1) 
        den_sum_f_frac = np.sum(f_frac, axis=1)
        aver_f_elecneg = np.divide(num_aver_f_elecneg, den_sum_f_frac, out=np.zeros_like(num_aver_f_elecneg), where=den_sum_f_frac != 0) 
        aver_f_elecneg_df = pd.DataFrame(aver_f_elecneg, columns=['aver_f_elecneg']) 
        #ic(aver_f_elecneg_df)

        #Calculate standard deviation electronegativity of each system (anions, cations and fillers)
        num_dev_a_elecneg = np.sum(a_frac*(a_elecneg-aver_a_elecneg[:, np.newaxis])**2, axis=1)
        dev_a_elecneg = np.sqrt(num_dev_a_elecneg / den_sum_a_frac)
        dev_a_elecneg_df = pd.DataFrame(dev_a_elecneg, columns=['dev_a_elecneg'])
        #ic(dev_a_elecneg_df)

        num_dev_c_elecneg = np.sum(c_frac*(c_elecneg-aver_c_elecneg[:, np.newaxis])**2, axis=1)
        dev_c_elecneg = np.sqrt(num_dev_c_elecneg / den_sum_c_frac)
        dev_c_elecneg_df = pd.DataFrame(dev_c_elecneg, columns=['dev_c_elecneg'])
        #ic(dev_c_elecneg_df)

        num_dev_f_elecneg = np.sum(f_frac*(f_elecneg-aver_f_elecneg[:, np.newaxis])**2, axis=1)
        dev_f_elecneg = np.sqrt(num_dev_f_elecneg / den_sum_f_frac, out=np.zeros_like(num_aver_f_elecneg), where=den_sum_f_frac != 0)
        dev_f_elecneg_df = pd.DataFrame(dev_f_elecneg, columns=['dev_f_elecneg'])
        #ic(dev_f_elecneg_df)

        #Calculate weighted average ionization energy of each system
        num_aver_a_ip = np.sum(a_ip * a_frac, axis=1) 
        aver_a_ip = np.divide(num_aver_a_ip, den_sum_a_frac) 
        aver_a_ip_df = pd.DataFrame(aver_a_ip, columns=['aver_a_ip']) 
        #ic(aver_a_ip_df)

        num_aver_c_ip = np.sum(c_ip * c_frac, axis=1) 
        aver_c_ip = np.divide(num_aver_c_ip, den_sum_c_frac) 
        aver_c_ip_df = pd.DataFrame(aver_c_ip, columns=['aver_c_ip']) 
        #ic(aver_c_ip_df)

        num_aver_f_ip = np.sum(f_ip * f_frac, axis=1)
        aver_f_ip = np.divide(num_aver_f_ip, den_sum_f_frac, out=np.zeros_like(num_aver_f_ip), where=den_sum_f_frac != 0) 
        aver_f_ip_df = pd.DataFrame(aver_f_ip, columns=['aver_f_ip'])
        #ic(aver_f_ip_df)

        #Calculate stand. dev. ionization energy for each system
        num_dev_a_ip = np.sum(a_frac*(a_ip-aver_a_ip[:, np.newaxis])**2, axis=1)
        dev_a_ip = np.sqrt(num_dev_a_ip / den_sum_a_frac)
        dev_a_ip_df = pd.DataFrame(dev_a_ip, columns=['dev_a_ip'])
        #ic(dev_a_ip_df)

        num_dev_c_ip = np.sum(c_frac*(c_ip-aver_c_ip[:, np.newaxis])**2, axis=1)
        dev_c_ip = np.sqrt(num_dev_c_ip / den_sum_c_frac)
        dev_c_ip_df = pd.DataFrame(dev_c_ip, columns=['dev_c_ip'])
        #ic(dev_c_ip_df)

        num_dev_f_ip = np.sum(f_frac*(f_ip-aver_f_ip[:, np.newaxis])**2, axis=1)
        dev_f_ip = np.sqrt(num_dev_f_ip / den_sum_f_frac, out=np.zeros_like(num_aver_f_ip), where=den_sum_f_frac != 0)
        dev_f_ip_df = pd.DataFrame(dev_f_ip, columns=['dev_f_ip'])
        #ic(dev_f_ip_df)

        #Calculate weighted average electron affinity for each system
        num_aver_a_ea = np.sum(a_ea * a_frac, axis=1) 
        aver_a_ea = np.divide(num_aver_a_ea, den_sum_a_frac) 
        aver_a_ea_df = pd.DataFrame(aver_a_ea, columns=['aver_a_ea']) 
        #ic(aver_a_ea_df)

        num_aver_c_ea = np.sum(c_ea * c_frac, axis=1) 
        aver_c_ea = np.divide(num_aver_c_ea, den_sum_c_frac) 
        aver_c_ea_df = pd.DataFrame(aver_c_ea, columns=['aver_c_ea']) 
        #ic(aver_c_ea_df)

        num_aver_f_ea = np.sum(f_ea * f_frac, axis=1)
        aver_f_ea = np.divide(num_aver_f_ea, den_sum_f_frac, out=np.zeros_like(num_aver_f_ea), where=den_sum_f_frac != 0) 
        aver_f_ea_df = pd.DataFrame(aver_f_ea, columns=['aver_f_ea'])
        #ic(aver_f_ea_df)

        #Calculate stand. dev. electron affinity
        num_dev_a_ea = np.sum(a_frac*(a_ea-aver_a_ea[:, np.newaxis])**2, axis=1)
        dev_a_ea = np.sqrt(num_dev_a_ea / den_sum_a_frac)
        dev_a_ea_df = pd.DataFrame(dev_a_ea, columns=['dev_a_ea'])
        #ic(dev_a_ea_df)

        num_dev_c_ea = np.sum(c_frac*(c_ea-aver_c_ea[:, np.newaxis])**2, axis=1)
        dev_c_ea = np.sqrt(num_dev_c_ea / den_sum_c_frac)
        dev_c_ea_df = pd.DataFrame(dev_c_ea, columns=['dev_c_ea'])
        #ic(dev_c_ea_df)

        num_dev_f_ea = np.sum(f_frac*(f_ea-aver_f_ea[:, np.newaxis])**2, axis=1)
        dev_f_ea = np.sqrt(num_dev_f_ea / den_sum_f_frac, out=np.zeros_like(num_aver_f_ea), where=den_sum_f_frac != 0)
        dev_f_ea_df = pd.DataFrame(dev_f_ea, columns=['dev_f_ea'])
        #ic(dev_f_ea_df)

        #Calculate valence electrons of each compound (row)
        total_val_e = np.sum(frac * val_e, axis=1) 
        total_val_e_df = pd.DataFrame(total_val_e, columns=['total_val_e']) 
        #ic(total_val_e_df)

        #Calculate mass of each compound
        total_mass = np.sum(frac * mass, axis=1)
        total_mass_df = pd.DataFrame(total_mass, columns=['total_mass'])
        #ic(total_mass_df)
        
        #Calculate mass of fillers
        f_mass = np.sum(f_frac * fillers_mass, axis=1)
        f_mass_df = pd.DataFrame(f_mass, columns=['f_mass'])
        #ic(f_mass_df)
            
        ###################################################################################################################
        #### New feature engineering - Std.Dev. mass (f,a,c) and frac (f,a,c) / Sum frac (f,a,c) & Ratio a_frac/c_frac ####
        ###################################################################################################################
        ic(df.columns)
        ### Std.Dev. mass (fillers, anions, cations)
        # Replace zeroes with NaN to ignore them in the Std.Dev. calculation
        df[['f_comp_1_mass', 'f_comp_2_mass', 'f_comp_3_mass', 'f_comp_4_mass', 'f_comp_5_mass']] = df[['f_comp_1_mass', 'f_comp_2_mass', 'f_comp_3_mass', 'f_comp_4_mass', 'f_comp_5_mass']].replace(0, np.nan)
        df[['a_comp_1_mass', 'a_comp_2_mass', 'a_comp_3_mass']] = df[['a_comp_1_mass', 'a_comp_2_mass', 'a_comp_3_mass']].replace(0, np.nan)
        df[['c_comp_1_mass', 'c_comp_2_mass', 'c_comp_3_mass']] = df[['c_comp_1_mass', 'c_comp_2_mass', 'c_comp_3_mass']].replace(0, np.nan)
        # Calculate Std.Dev. across rows, but set std to 0 if only one non-NaN value exists
        std_dev_f_mass = df[['f_comp_1_mass', 'f_comp_2_mass', 'f_comp_3_mass', 'f_comp_4_mass', 'f_comp_5_mass']].std(axis=1, skipna=True)
        std_dev_a_mass = df[['a_comp_1_mass', 'a_comp_2_mass', 'a_comp_3_mass']].std(axis=1, skipna=True)
        std_dev_c_mass = df[['c_comp_1_mass', 'c_comp_2_mass', 'c_comp_3_mass']].std(axis=1, skipna=True)
        # Check for rows with only one non-NaN value and set the Std.Dev. to 0 for those rows
        std_dev_f_mass[df[['f_comp_1_mass', 'f_comp_2_mass', 'f_comp_3_mass', 'f_comp_4_mass', 'f_comp_5_mass']].notna().sum(axis=1) == 1] = 0
        std_dev_a_mass[df[['a_comp_1_mass', 'a_comp_2_mass', 'a_comp_3_mass']].notna().sum(axis=1) == 1] = 0
        std_dev_c_mass[df[['c_comp_1_mass', 'c_comp_2_mass', 'c_comp_3_mass']].notna().sum(axis=1) == 1] = 0
        ic(std_dev_c_mass)
        # Convert to dataframes
        std_dev_all_mass_df = pd.concat([std_dev_f_mass, std_dev_a_mass, std_dev_c_mass], axis=1)
        std_dev_all_mass_df.columns = ['std_dev_f_mass', 'std_dev_a_mass', 'std_dev_c_mass']


        ### Std.Dev. fractions (fillers, anions, cations)
        # Replace zeroes with NaN to ignore them in the Std.Dev. calculation
        df[['f_frac_1', 'f_frac_2', 'f_frac_3', 'f_frac_4', 'f_frac_5']] = df[['f_frac_1', 'f_frac_2', 'f_frac_3', 'f_frac_4', 'f_frac_5']].replace(0, np.nan)
        df[['a_frac_1', 'a_frac_2', 'a_frac_3']] = df[['a_frac_1', 'a_frac_2', 'a_frac_3']].replace(0, np.nan)
        df[['c_frac_1', 'c_frac_2', 'c_frac_3']] = df[['c_frac_1', 'c_frac_2', 'c_frac_3']].replace(0, np.nan)
        # Calculate Std.Dev. across rows, but set std to 0 if only one non-NaN value exists
        std_dev_f_frac = df[['f_frac_1', 'f_frac_2', 'f_frac_3', 'f_frac_4', 'f_frac_5']].std(axis=1, skipna=True)
        std_dev_a_frac = df[['a_frac_1', 'a_frac_2', 'a_frac_3']].std(axis=1, skipna=True)
        std_dev_c_frac = df[['c_frac_1', 'c_frac_2', 'c_frac_3']].std(axis=1, skipna=True)
        # Check for rows with only one non-NaN value and set the Std.Dev. to 0 for those rows
        std_dev_f_frac[df[['f_frac_1', 'f_frac_2', 'f_frac_3', 'f_frac_4', 'f_frac_5']].notna().sum(axis=1) == 1] = 0
        std_dev_a_frac[df[['a_frac_1', 'a_frac_2', 'a_frac_3']].notna().sum(axis=1) == 1] = 0
        std_dev_c_frac[df[['c_frac_1', 'c_frac_2', 'c_frac_3']].notna().sum(axis=1) == 1] = 0
        # Convert to dataframes
        std_dev_all_frac_df = pd.concat([std_dev_f_frac, std_dev_a_frac, std_dev_c_frac], axis=1)
        std_dev_all_frac_df.columns = ['std_dev_f_frac', 'std_dev_a_frac', 'std_dev_c_frac']


        ### Sum fractions (fillers, anions, cations)
        sum_f_frac = df[['f_frac_1', 'f_frac_2', 'f_frac_3', 'f_frac_4', 'f_frac_5']].sum(axis=1)
        sum_a_frac = df[['a_frac_1', 'a_frac_2', 'a_frac_3']].sum(axis=1)
        sum_c_frac = df[['c_frac_1', 'c_frac_2', 'c_frac_3']].sum(axis=1)
        # Convert to dataframes
        sum_all_frac_df = pd.concat([sum_f_frac, sum_a_frac, sum_c_frac], axis=1)
        sum_all_frac_df.columns = ['sum_f_frac', 'sum_a_frac', 'sum_c_frac']


        ### Ratio a_frac/c_frac
        ratio_a_c = sum_all_frac_df['sum_a_frac'].div(sum_all_frac_df['sum_c_frac'])
        # Convert to dataframe
        ratio_a_c_df = pd.DataFrame(ratio_a_c, columns=['ratio_a_c'])
        ic(ratio_a_c_df)
        ###############################################################################################

        # Print num_anions, num_cations, num_fillers
        num_atoms_df = pd.DataFrame(num_atoms, columns=['num_anions', 'num_cations','num_fillers'])
        #ic(num_atoms_df)
        # Print all frac
        #frac_df = pd.DataFrame(frac, columns=['a_frac_1', 'a_frac_2', 'a_frac_3', 'c_frac_1', 'c_frac_2', 'c_frac_3', 'f_frac_1', 'f_frac_2', 'f_frac_3', 'f_frac_4', 'f_frac_5'])
        #ic(frac_df)
        # Print T
        t_df = pd.DataFrame(T, columns=['T'])
        #ic(t_df)
        # Print n_300
        n_300_df = pd.DataFrame(df.n_300, columns=['n_300'])
        #ic(n_300_df)
        # Print PF (labels)
        target_df = pd.DataFrame(target, columns=[raw_label])
        #ic(target_df)
        # Print n-p type
        p_n_df = pd.DataFrame(df.p_n, columns=['p_n'])
        # Summary descriptors PHASE 2
        df_descriptors_phase2 = pd.concat([num_atoms_df,
                                            aver_a_elecneg_df, dev_a_elecneg_df,
                                            aver_c_elecneg_df, dev_c_elecneg_df,
                                            aver_f_elecneg_df, dev_f_elecneg_df,
                                            aver_a_ip_df, dev_a_ip_df,
                                            aver_c_ip_df, dev_c_ip_df,
                                            aver_f_ip_df, dev_f_ip_df,
                                            aver_a_ea_df, dev_a_ea_df,
                                            aver_c_ea_df, dev_c_ea_df,
                                            aver_f_ea_df, dev_f_ea_df,
                                            f_mass_df, total_mass_df,
                                            std_dev_all_mass_df, # New features
                                            std_dev_all_frac_df, # New features
                                            sum_all_frac_df, # New features
                                            ratio_a_c_df, # New feature
                                            total_val_e_df,
                                            #frac_df, # Ignored (not used anymore...)
                                            n_300_df,
                                            t_df,
                                            p_n_df,
                                            target_df
                                            ], axis=1)
        ic(df_descriptors_phase2)

        descriptors_phase2_list = list(df_descriptors_phase2.columns)
        #ic(descriptors_phase2_list)
        # Descriptor matrix for uploaded systems. No file is written here,
        # which avoids conflicts between simultaneous web requests.

        ###################################################################
        #### Feature scaling using saved training scaler - PHASE 3       ####
        ###################################################################

        target = target_col
        dataset = df_descriptors_phase2.copy()
        features = dataset.drop(target, axis=1)
        features['n_300'].fillna(0, inplace=True) # Replace NaN with zeros - New in this v5, didn't think about this until latest test sets 
        features['std_dev_f_mass'].fillna(0, inplace=True)
        features['std_dev_f_frac'].fillna(0, inplace=True)
        ic(features.columns)
        label = dataset[target]

        # Load scaler and feature order from training
        print(f"* Loading scaler from: {SCALER_FILE}")
        scaler = load(SCALER_FILE)

        print(f"* Loading feature order from: {FEATURES_FILE}")
        feature_columns = load_feature_order(FEATURES_FILE)

        # Ensure we have exactly the same features in the same order
        missing = set(feature_columns) - set(features.columns)
        extra   = set(features.columns) - set(feature_columns)

        if missing:
            raise ValueError(f"Missing features in unknown data: {missing}")
        # Extra columns can be ignored; we just select the training ones
        features = features[feature_columns]
        ic(features)

        X_unknown = features.to_numpy()
        ic(X_unknown)

        # 3) Scale ALL unknown systems in one shot (NO fit here!)
        print("* Scaling unknown systems using training scaler (transform only)...")
        X_unknown_scaled = scaler.transform(X_unknown)
        X_unknown_scaled = X_unknown_scaled[:, np.newaxis, :]

        all_norm_samples = X_unknown_scaled
        ic(all_norm_samples)

    #import sys
    #sys.exit('Testing ended!')
    
    
    print('--------------------------------------------')
    print('* Collecting normalised samples...')
    '''import sys
    sys.exit('Testing ended!')'''


    # Model path
    saved_model = 'model_keras_skutt.h5'
    print(f'* Loading model saved in file: {saved_model}')
    # Loading model
    print('* Loading and compiling model...')
    model = load_model(saved_model, compile=True,
                       custom_objects={'LeakyReLU': LeakyReLU,
                                       'mse': MeanSquaredError(),
                                       },
                       )

    # Iterate through samples to be predicted and generate predictions for samples
    all_predicted = []
    all_expected = []
    print('* Predicting...')
    count = 0
    ic(all_norm_samples)
    for norm_sample in all_norm_samples:
        ic(norm_sample)
        ic(norm_sample.shape)
        #import sys
        #sys.exit('Testing ended!')
        count +=1
        prediction = model.predict(norm_sample, verbose=0)
        # Just aesthetics :P print only the number and not like an ugly NumPy array
        #prediction_str = str(prediction)
        #prediction_str = prediction_str.replace('[','')
        #prediction_str = prediction_str.replace(']','')
        prediction_item = prediction.item()
        ic(prediction_item)
        print(f'   ++ Predicted {target} value for sample {count}: {prediction_item}')
        all_predicted.append(float(prediction_item))

    return all_predicted
