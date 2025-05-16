import matplotlib.pyplot as plt
from fooof.sim.gen import gen_power_spectrum
import pandas as pd
import numpy as np
import os

input_folder_electrodes = "Z:/LeoF/Cygnus/Electrode"
input_folder_power = "Z:/LeoF/Cygnus/Files"
exps = pd.read_csv('z:/LeoF/Cygnus/DataFrame/data_frame.csv')

def extract_info(filename):
    parts = filename.split('_')
    return parts[1][0], parts[2] + "_" + parts[3]

def filter_freq_spectra(xf, spectra):
    xf_positive = xf[xf >= 0]
    mask = (xf_positive > 3) & (xf_positive < 26)
    xf_filtered = xf_positive[mask]
    n_positive = len(xf_positive)
    spectra_positive = spectra[:, :n_positive]
    spectra_filtered = spectra_positive[:, mask]
    return xf_filtered, spectra_filtered.mean(axis = 0 )

def model_1(pre_event_spectra, post_event_spectra):
    return post_event_spectra - pre_event_spectra

def model_2(pre_event_spectra_filtered, post_event_spectra_filtered):
    fft_erp = power_data[:, 3, :]
    _, fft_erp_filtered = filter_freq_spectra(xf_post, fft_erp)
    model_2 = pre_event_spectra_filtered + fft_erp_filtered
    return post_event_spectra_filtered - model_2

def model_3(pre_event_spectra_filtered, post_event_spectra_filtered):
    """
    1/f components reconstructued based off offset and slope (differences) --
    prob between pre and post + 
    """
    fft_erp = power_data[:, 3, :]
    _, fft_erp_filtered = filter_freq_spectra(xf_post, fft_erp)

    exps = pd.read_csv('z:/LeoF/Cygnus/DataFrame/data_frame.csv')

    exps_pre = exps[exps['Time Window'] == 'yf_avg_pre'] # Pre exps
    exps_post = exps[exps['Time Window'] == 'yf_avg_post'] # Post exps

    exps_pre = exps_pre.sort_values(by=['Participant', 'Electrode', 'Condition']).reset_index()
    exps_post = exps_post.sort_values(by=['Participant', 'Electrode', 'Condition']).reset_index()

    compared = exps_pre[['Participant', 'Electrode', 'Condition']].compare(other = exps_post[['Participant', 'Electrode', 'Condition']], keep_equal=False) 
    assert compared.empty == True # Should be empty if no mismatches

    # Construcing delta offset and exponent
    exps_post['post_minus_pre_Exponent'] = (exps_post['Exponent'] - exps_pre['Exponent']) # Compute post-pre differences
    exps_post['post_minus_pre_Offset'] = (exps_post['Offset'] - exps_pre['Offset'])

    # Averaging across electrodes so we get participant and condition as the dimensions (e.g. pip types)
    delta_averages = (
        exps_post
            .groupby(['Participant','Condition'])
            .mean(['post_minus_pre_Exponent','post_minus_pre_Offset'])
            .reset_index()
    )
    pre_averages = (
        exps_pre
            .groupby(['Participant','Condition'])[['Offset','Exponent']]
            .mean()
            .reset_index()
    )

    # Acquiring correct participant and condition 
    participant = trial_power_file.split('.')[0]
    condition = '_'.join(trial_power_file.split('_')[2:4])

    # Indexing w. Particpant and Condition vars
    delta_condition = delta_averages[
        (delta_averages['Participant'] == participant) &
        (delta_averages["Condition"] == condition)
        ]
    
    participant_pre = pre_averages[
        (pre_averages['Participant'] == participant) &
        (pre_averages['Condition']   == condition)
        ]
    
    delta_exponent = delta_condition['post_minus_pre_Exponent'].iloc[0]
    delta_offset = delta_condition['post_minus_pre_Offset'].iloc[0]

    pre_offset   = participant_pre['Offset'].iloc[0]
    pre_exponent = participant_pre['Exponent'].iloc[0]  

    # Building absolute aperiodic components
    absolute_offset   = pre_offset   + delta_offset
    absolute_exponent = pre_exponent + delta_exponent
    aperiodic_params  = [absolute_offset, absolute_exponent]

    xf_positive = xf_pre[xf_pre >= 0]
    mask = (xf_positive > 3) & (xf_positive < 26)
    xf_filtered = xf_positive[mask]

    _, sim_power= gen_power_spectrum(
        freq_range = [xf_filtered[0], xf_filtered[-1]],
        aperiodic_params = aperiodic_params,
        periodic_params = [],
        nlv = 0,
        freq_res = 1.6623376623376624 #hard coded -- calculated in another script 
    )

    model_3 = pre_event_spectra_filtered + fft_erp_filtered + sim_power
    return post_event_spectra_filtered - model_3

# if I get the power by condition I am fonna have to do a matching for loop 
# if if get by just condition I think I have to do a lot more coding 
residual_powers_model_1 = []
residual_powers_model_2 = []
residual_powers_model_3 = []

for trial_freq_file in os.listdir(input_folder_power):
    if not trial_freq_file.endswith("freq.npy"):
        continue  # Skip non-freq files

    freq_num, freq_pip_type = extract_info(trial_freq_file)

    for trial_power_file in os.listdir(input_folder_power):
        if not trial_power_file.endswith("power.npy"):
            continue  # Skip non-power files

        power_num, power_pip_type = extract_info(trial_power_file)

        if freq_num == power_num and freq_pip_type == power_pip_type:
            freq_data = np.load(os.path.join(input_folder_power, trial_freq_file), allow_pickle=True)
            power_data = np.load(os.path.join(input_folder_power, trial_power_file), allow_pickle=True)

            pre_event_spectra = power_data[:, 0, :]
            post_event_spectra = power_data[:, 1, :]

            freq_data = freq_data.item()
            xf_pre = freq_data['xf_pre']
            xf_post = freq_data['xf_post']

            xf_pre_filtered, pre_event_spectra_filtered = filter_freq_spectra(xf_pre, pre_event_spectra)
            xf_post_filtered, post_event_spectra_filtered = filter_freq_spectra(xf_post, post_event_spectra)

            residual_power_model_1 = model_1(pre_event_spectra_filtered, post_event_spectra_filtered)
            residual_powers_model_1.append(residual_power_model_1)

            residual_power_model_2 = model_2(pre_event_spectra_filtered, post_event_spectra_filtered)
            residual_powers_model_2.append(residual_power_model_2)
            residual_power_model_3 = model_3(pre_event_spectra_filtered, post_event_spectra_filtered)
            residual_powers_model_3.append(residual_power_model_3)


plt.plot(xf_post_filtered, residual_power_model_1, label="Model 1")
plt.plot(xf_post_filtered, residual_power_model_2, label="Model 2")
plt.plot(xf_post_filtered, residual_power_model_3, label = "Model 3")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Residual Power")
plt.legend()
plt.show()


# fft_erp = power_data[:, 3, :]
# _, fft_erp_filtered = filter_freq_spectra(xf_post, fft_erp)

# exps = pd.read_csv('z:/LeoF/Cygnus/DataFrame/data_frame.csv')

# exps_pre = exps[exps['Time Window'] == 'yf_avg_pre'] # Pre exps
# exps_post = exps[exps['Time Window'] == 'yf_avg_post'] # Post exps

# exps_pre = exps_pre.sort_values(by=['Participant', 'Electrode', 'Condition']).reset_index()
# exps_post = exps_post.sort_values(by=['Participant', 'Electrode', 'Condition']).reset_index()

# compared = exps_pre[['Participant', 'Electrode', 'Condition']].compare(other = exps_post[['Participant', 'Electrode', 'Condition']], keep_equal=False) 
# assert compared.empty == True # Should be empty if no mismatches

# exps_post['post_minus_pre_Exponent'] = (exps_post['Exponent'] - exps_pre['Exponent']) # Compute post-pre differences
# exps_post['post_minus_pre_Offset'] = (exps_post['Offset'] - exps_pre['Offset'])

# condition_averages = exps_post.groupby(['Participant','Condition']).mean(['post_minus_pre_Exponent','post_minus_pre_Offset']).reset_index()
# participant = trial_power_file.split('.')[0]
# condition = '_'.join(trial_power_file.split('_')[2:4])
# participant_condition = condition_averages[(condition_averages['Participant'] == participant) & (condition_averages["Condition"] == condition)]
# delta_exponent = participant_condition['post_minus_pre_Exponent'].iloc[0]
# delta_offset = participant_condition['post_minus_pre_Offset'].iloc[0]
# aperiodic_params = [delta_offset, delta_exponent]

# # Filtering out frequencies outside the range that we care about 
# xf_positive = xf_pre[xf_pre >= 0]
# mask = (xf_positive > 3) & (xf_positive < 26)
# xf_filtered = xf_positive[mask]

# #
# _, sim_power= gen_power_spectrum(
# freq_range = [xf_filtered[0], xf_filtered[-1]],
# aperiodic_params = aperiodic_params,
# periodic_params = [],
# nlv = 0,
# freq_res = 1.6623376623376624 #hard coded -- calculated in another script 
# )


# # plt.plot(xf_post_filtered, pre_event_spectra_filtered, label="pre_event")
# plt.plot(xf_post_filtered, fft_erp_filtered, label = "erp")
# # plt.plot(xf_post_filtered, sim_power, label = 'aperiodic')
# plt.show()

#pre stimulus exponent  + exponent delta
# then remove the pre stimulus exponent when calculating the model_3


# Re-accounting steps for computing model 3 residuals: 

# Dillan correct me if I am wrong but the option b for computing model 3 was to simulate the
