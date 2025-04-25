import numpy as np 
import pandas as pd
import scipy.stats
import os

n_permutations = 10000 # Number of permutations for statistical testing
n_swap = 0.5 # Fraction of data to shuffle 

df_dir = "Z:/LeoF/Cygnus/DataFrame/"

exps = pd.read_csv(df_dir+"data_frame.csv")

exps_post_minus_erp = exps[exps['Time Window'] == 'post_minus_erp']

exps_post_minus_erp = exps_post_minus_erp.sort_values(by=['Participant', 'Electrode', 'Condition']).reset_index()

compared = exps_pre[['Participant', 'Electrode', 'Condition']].compare(other = exps_post_minus_erp[['Participant', 'Electrode', 'Condition']], keep_equal=False) 
print(compared) # Ensuring no mismatches
assert compared.empty == True # Should be empty if no mismatches

exps_post_minus_erp['pre_Rsq'] = exps_pre['Rsq']
cleaned_post_minus_erp = exps_post_minus_erp[(exps_post_minus_erp['Rsq'] >= 0.9)]

grpby_post_minus_erp = cleaned_post_minus_erp.groupby(['Participant','Condition']).mean(['Offset','Rsq']).reset_index() 

permuted_post_minus_erp_df = pd.DataFrame()
for condition in ['frequent_rare', 'frequent_frequent', 'rare_frequent']:

    temp_dict = {'condition' : ([condition]*n_permutations)} # E.g. condition = 'frequent_rare', n_permutations

    grpby_this_cond = grpby_post_minus_erp[grpby_post_minus_erp['Condition'] == condition] # Filter for rows of current conditon 
    n_rows = len(grpby_this_cond) # 20 rows
    n_rows_to_shuffle = int( n_rows*n_swap) # shuffle half of subjects (20* 0.5 = 10)

    exps_intact = grpby_this_cond['Exponent'].values # Return np arrays
    offs_intact = grpby_this_cond['Offset'].values

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
    permuted_post_minus_erp_df  = pd.concat([temp_df, permuted_post_minus_erp_df ], ignore_index=True)

    print(f"Condition {condition} done!")

permuted_post_minus_erp_df .to_csv("Z:\LeoF\Cygnus\DataFrame\permutation_post_minus_erp_shuffled.csv")

# Compute grand mean of offest & exponent
condition = [ 'frequent_rare', 'rare_frequent', 'frequent_frequent']

# Filter by condition 
for i in condition:
    
    # Calculating percentile by condition
    exp_percentiles = np.percentile(permuted_post_minus_erp_df[permuted_post_minus_erp_df['condition'] == i]['null_exponents'], [2.5, 97.5])
    off_percentiles = np.percentile(permuted_post_minus_erp_df[permuted_post_minus_erp_df['condition'] == i]['null_offsets'], [2.5, 97.5])
    
    # Calculating exponent by condition
    exp_grand_average = np.mean(cleaned_post_minus_erp[cleaned_post_minus_erp['Condition'] == i]['Exponent'])
    off_grand_average = np.mean(cleaned_post_minus_erp[cleaned_post_minus_erp['Condition'] == i]['Offset'])
    
    # Determining significance
    if (exp_grand_average < exp_percentiles[0] or exp_grand_average > exp_percentiles[1]):
        print(f"The exponent grand mean for {i} of {exp_grand_average:.2f} is significant!")
    else:
        print(f"The exponent grand mean for {i} of {exp_grand_average:.2f} is not significant.")
    if (off_grand_average < off_percentiles[0] or off_grand_average > off_percentiles[1]):
        print(f"The offset grand mean for {i} of {off_grand_average:.2f} is significant!")
    else:
        print(f"The exponent grand mean for {i} of {off_grand_average:.2f} is not significant.")