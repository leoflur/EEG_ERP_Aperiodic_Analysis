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
                baseline=None, preload=True
            )
            
            conditions = ['frequent_rare', 'rare_frequent', 'frequent_frequent']
            electrodes = epochs.ch_names

            #adds an empty dictionary, keyed by the filename (e.g. {'p3_1': {}})
            # y_frequencies_dict[filename.split('.')[0]] = {}
            
            
            for electrode in electrodes:  
                for condition in conditions:
                    # 5. Separate Epochs into Pre/Post (inlined from separate_epochs)
                    epochs_cond = epochs[condition].copy().pick_channels([electrode])
                    epochs_pre = epochs_cond.copy().crop(tmin=-0.6, tmax=0)
                    epochs_post = epochs_cond.copy().crop(tmin=0, tmax=0.6)
                    
                    # Debug: Check epoch counts
                    print(f"Condition {condition}: {len(epochs_pre)} pre and {len(epochs_post)} post epochs")
                    assert len(epochs_pre) == len(epochs_post)
                    
                    # 6. Perform FFT on Pre/Post (inlined from perform_fft_on_epochs)
                    pre_data = epochs_pre.get_data()[:, 0, :]  # Shape: (n_epochs, n_samples)
                    post_data = epochs_post.get_data()[:, 0, :]
                    
                    pre_spectra = np.abs(fft(pre_data, axis=1))
                    post_spectra = np.abs(fft(post_data, axis=1))
                    
                    yf_avg_pre = pre_spectra.mean(axis=0)
                    yf_avg_post = post_spectra.mean(axis=0)
                    
                    # 7. Compute ERP (inlined from compute_erp)
                    erp = epochs_post.average()  # Fix: Use epochs_post directly
                    
                    # 8. FFT on ERP (inlined from perform_fft_on_erp)
                    erp_data = erp.pick([electrode]).get_data()[0, :]  # Shape: (n_samples,)
                    fft_erp = np.abs(fft(erp_data))
                    
                    # 9. Get Frequency Axes (inlined from get_xf)
                    sample_rate = epochs_pre.info['sfreq']
                    N_pre = epochs_pre.get_data().shape[-1]
                    xf = fftfreq(N_pre, 1/sample_rate)
                    
                    N_erp = len(erp.times)
                    xf_erp = fftfreq(N_erp, 1/sample_rate)
                    
                    # Debug: Check frequency axis shapes
                    assert yf_avg_pre.shape == xf.shape
                    assert fft_erp.shape == xf_erp.shape
                    
                    # 10. Post-minus-ERP Calculation
                    post_minus_erp = yf_avg_post - fft_erp
                    
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
    y_frequencies_dict = process_eeg_files(folder_path)

print(y_frequencies_dict)