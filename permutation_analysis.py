

import numpy as np 
import pandas as pd
import scipy.stats
import os

n_permutations = 10000 # Number of permutations for statistical testing
n_swap = 0.5 # Fraction of data to shuffle 

df_dir = "Z:/LeoF/Cygnus/DataFrame/"

exps = pd.read_csv(df_dir+"data_frame.csv")

exps_pre = exps[exps['Time Window'] == 'yf_avg_pre'] # Pre exps
exps_post = exps[exps['Time Window'] == 'yf_avg_post'] # Post exps

exps_pre = exps_pre.sort_values(by=['Participant', 'Electrode', 'Condition']).reset_index()
exps_post = exps_post.sort_values(by=['Participant', 'Electrode', 'Condition']).reset_index()

compared = exps_pre[['Participant', 'Electrode', 'Condition']].compare(other = exps_post[['Participant', 'Electrode', 'Condition']], keep_equal=False) 
print(compared) # Ensuring no mismatches
assert compared.empty == True # Should be empty if no mismatches

exps_post['post_minus_pre_Exponent'] = (exps_post['Exponent'] - exps_pre['Exponent']) # Compute post-pre differences
exps_post['post_minus_pre_Offset'] = (exps_post['Offset'] - exps_pre['Offset'])

grpby = exps_post.groupby(['Participant','Condition']).mean(['post_minus_pre_Exponent','post_minus_pre_Offset']).reset_index() # Aggregate data

### On each permutation, and within each condition
### 1) get the mean diff for each params, avg across electrodes, one point each subject
### 2) take half of the rows, and flip the sign of the param diff
### 3) now average across all subjects to get a grand mean for this perm
### 4) save as one row of our dictionary

permuted_df = pd.DataFrame()
for condition in ['frequent_rare', 'frequent_frequent', 'rare_frequent']:

    temp_dict = {'condition' : ([condition]*n_permutations)} # E.g. condition = 'frequent_rare', n_permutations

    grpby_this_cond = grpby[grpby['Condition'] == condition] # Filter for rows of current conditon 
    n_rows = len(grpby_this_cond) # 20 rows
    n_rows_to_shuffle = int( n_rows*n_swap) # shuffle half of subjects (20* 0.5 = 10)

    exps_intact = grpby_this_cond['post_minus_pre_Exponent'].values # Return np arrays
    offs_intact = grpby_this_cond['post_minus_pre_Offset'].values

    perms_exps = []
    perms_offs = []
    for p in range(n_permutations): # 1000 iterations

        perms_idx = np.random.permutation(n_rows) # Permute len index of contiion
        perms_idx_to_flip = perms_idx[:n_rows_to_shuffle]# Get first half
        perms_idx_to_leave = perms_idx[n_rows_to_shuffle: ] # Get second half

        exps_sign_flip = (exps_intact[perms_idx_to_flip]*-1)
        exps_no_sign_flip = exps_intact[perms_idx_to_leave]
        permuted_exp = np.mean(np.hstack([exps_sign_flip, exps_no_sign_flip]))

        offs_sign_flip = (offs_intact[perms_idx_to_flip]*-1) # Filter by array of indices (this is legal apparently)
        offs_no_sign_flip = offs_intact[perms_idx_to_leave]
        permuted_off = np.mean(np.hstack([offs_sign_flip, offs_no_sign_flip]))

        perms_exps.append(permuted_exp)
        perms_offs.append(permuted_off)
    
    temp_dict['null_exponents'] = perms_exps
    temp_dict['null_offsets'] = perms_offs
    temp_dict['n_permutation'] = list(range(n_permutations))

    temp_df = pd.DataFrame().from_dict(temp_dict, orient='index').transpose()
    permuted_df = pd.concat([temp_df, permuted_df], ignore_index=True)

    print(f"Condition {condition} done!")

permuted_df.to_csv(os.getcwd()+'\\files\\permutation_shuffled.csv')
