import mne
import matplotlib.pyplot as plt
import numpy as np 
import os
from itertools import chain
from scipy.fft import fft, fftfreq
from fooof import FOOOF
import seaborn as sns
import pandas as pd

def process_eeg_files(folder_path):
    data_frame = pd.DataFrame()
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.set'):
            file_path = os.path.join(folder_path, filename)
            
            # 1. Load Raw EEG Data
            raw_eeg = mne.io.read_raw_eeglab(file_path, preload=True)
            
            # 2. Preprocess Raw EEG (inlined from preprocess_raw_eeg)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw_eeg.rename_channels({"FP1": 'Fp1', 'FP2': 'Fp2'})
            raw_eeg.set_channel_types({
                'HEOG_left': 'eog', 'HEOG_right': 'eog',
                'VEOG_lower': 'eog', '(uncorr) HEOG': 'eog',
                '(uncorr) VEOG': 'eog'
            })
            raw_eeg.set_montage(montage)
            
            # 3. Extract and Modify Events (inlined from extract_and_modify_events)
            events, _ = mne.events_from_annotations(raw_eeg)
            events = events[~np.isin(events[:, 2], [6, 7])]
            
            target_ids = [1, 9, 15, 21, 27]
            nontarget_ids = [2, 3, 4, 5, 8, 10, 11, 12, 13, 14,
                            16, 17, 18, 19, 20, 22, 23, 24, 25, 26]
            
            for i in range(len(events)):
                current_id = events[i, 2]
                if current_id in target_ids:
                    events[i, 2] = 100  # 'frequent_rare'
                elif (i > 0 and current_id in nontarget_ids and
                      events[i - 1, 2] == 100):
                    events[i, 2] = 200  # 'rare_frequent'
                elif (i > 0 and current_id in nontarget_ids and
                      events[i - 1, 2] in [200, 300]):
                    events[i, 2] = 300  # 'frequent_frequent'
            
            event_id = {
                "frequent_rare": 100,
                "rare_frequent": 200,
                "frequent_frequent": 300
            }
            
            # 4. Create Epochs (inlined from create_epochs)
            epochs = mne.Epochs(
                raw_eeg, events=events, event_id=event_id,
                tmin=-0.6, tmax=0.6, 
                picks = "eeg",
                baseline= (-0.6,0), preload=True
            )
            
            conditions = ['frequent_rare', 'rare_frequent', 'frequent_frequent']
            electrodes = epochs.ch_names

            #adds an empty dictionary, keyed by the filename (e.g. {'p3_1': {}})
            # y_frequencies_dict[filename.split('.')[0]] = {}
            
            
            for electrode in electrodes:  
                for condition in conditions:
                    # # 5. Separate Epochs into Pre/Post (inlined from separate_epochs)
                    # epochs_cond = epochs[condition].copy().pick_channels([electrode])
                    # epochs_pre = epochs_cond.copy().crop(tmin=-0.6, tmax=0)
                    # epochs_post = epochs_cond.copy().crop(tmin=0, tmax=0.6)
                    
                    # # Debug: Check epoch counts
                    # print(f"Condition {condition}: {len(epochs_pre)} pre and {len(epochs_post)} post epochs")
                    # assert len(epochs_pre) == len(epochs_post)
                    
                    # # 6. Perform FFT on Pre/Post (inlined from perform_fft_on_epochs)
                    # pre_data = epochs_pre.get_data()[:, 0, :]  # Shape: (n_epochs, n_samples)
                    # post_data = epochs_post.get_data()[:, 0, :]

                    # Below is a an improved approach where we use boolean masking instead of having to create multiple copeis all the time
                    # Get electrodes for the condition
                    epochs_data = epochs[condition].get_data() # Shape : ( n_epocs, n_channels, n_times)
                    times = epochs.times # Time vector from original epochs

                    # Find electrode index
                    electrode_idx = epochs.ch_names.index(electrode)

                    # Extract data for this electrode
                    electrode_data = epochs_data[:,electrode_idx, : ]
                    
                    # Create time masks
                    pre_mask = (times >= -0.6) & (times <= 0)
                    post_mask = (times >= 0) & (times <= 0.6)

                    # Split into pre and post
                    pre_data = electrode_data[:, pre_mask]  # Shape: (n_epochs, n_pre_samples)
                    post_data = electrode_data[:, post_mask]

                    print(f"Condition {condition}: {pre_data.shape[0]} pre and {post_data.shape[0]} post epochs")
                    assert pre_data.shape[0] == post_data.shape[0]
                    
                    # Compute FFT on pre/post data
                    pre_spectra = np.abs(np.fft.fft(pre_data, axis=1))  # FFT along time axis
                    post_spectra = np.abs(np.fft.fft(post_data, axis=1))
                    
                    # Average spectra across epochs
                    yf_avg_pre = pre_spectra.mean(axis=0)
                    yf_avg_post = post_spectra.mean(axis=0)
                    
                    # Compute ERP (average across epochs)
                    erp = post_data.mean(axis = 0 ) # Shape: (n_post_samples,)
                    
                    # FFT on ERP (inlined from perform_fft_on_erp)
                    fft_erp = np.abs(np.fft.fft(erp))

                    # Get frequency axis (using origina sampling rate)
                    sample_rate = epochs.info['sfreq'] # Get SF from original epochs
                    N_erp = pre_data.shape[1] # Number of time samples in pre
                    xf = np.fft.fftfreq(N_erp, 1 / sample_rate)
                    
                    N_erp = post_data.shape[1] # Number of time sampels in post-data
                    xf_erp = fftfreq(N_erp, 1/sample_rate)
                    
                    # Debug: Check frequency axis shapes
                    assert yf_avg_pre.shape == xf.shape
                    assert fft_erp.shape == xf_erp.shape
                    
                    # 10. Post-minus-ERP Calculation
                    post_minus_erp = yf_avg_post - fft_erp

                    #need to add in a np.save here for all the power spectral data 
                    # compile them into 
                    
                    # 11. FOOOF Analysis
                    power_spec_windows = [
                        ("yf_avg_pre", yf_avg_pre),
                        ("yf_avg_post", yf_avg_post),
                        ("post_minus_erp", post_minus_erp)
                    ]
                    new_row = pd.DataFrame({
                            'Participant': [filename.split('.')[0]],
                            'Electrode': [electrode],
                            'Condition': [condition]
                        })

                    for name, power_spec in power_spec_windows:
                        fg = FOOOF(
                            peak_width_limits=[2.5, 8],
                            max_n_peaks=1,
                            min_peak_height=0.3,
                            peak_threshold=2.0,
                            aperiodic_mode='fixed'
                        )
                        
                        fg.fit(xf, power_spec, freq_range=(2, 25))

                        try:
                            fg.fit(xf, power_spec, freq_range=(2, 25))
                            if fg.has_model:
                                exp = fg.get_params('aperiodic_params')[1]
                                offset = fg.get_params('aperiodic_params')[0]
                                rsq = fg.get_params('r_squared')
                            else:
                                exp, offset, rsq = np.nan, np.nan, np.nan
                        except:
                            exp, offset, rsq = np.nan, np.nan, np.nan

                        #if fg.has_model is False give NaNs values otherwise keep script the same

                        new_row['Time Window'] = name
                        new_row['Exp'] =  fg.get_params('aperiodic_params')[1]  # Index 1 corresponds to the exponent
                        new_row['Offset'] =  fg.get_params('aperiodic_params')[0]  # Index 0 corresponds to the offset
                        new_row['Rsq'] = fg.get_params('r_squared')
                        
                        # results[name] = {
                        #     name: power_spec,
                        #     'aps': fg.get_params('aperiodic_params'),
                        #     'rsq': fg.get_params('r_squared'),
                        # }

                    data_frame = pd.concat([data_frame, new_row], ignore_index=True)
                    # results.update({
                    #     'fft_erp': fft_erp,
                    #     'xf': xf,
                    #     'xf_erp': xf_erp
                    # })
                    
                    # y_frequencies_dict[filename.split('.')[0]][condition] = results
    
    return data_frame

if __name__ == '__main__':
    folder_path = "C:/Users/leofl/OneDrive/Pictures/Documents/GitHub/Project-Cygnus/files"
    y_frequencies_df = process_eeg_files(folder_path)
    y_frequencies_df.to_csv("Z:/LeoF/Cygnus/y_frequencies_df.csv")
    # this is where np.save will go 
    
print(y_frequencies_df)