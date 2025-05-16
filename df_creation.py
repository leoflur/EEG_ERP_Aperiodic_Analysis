import mne
import matplotlib.pyplot as plt
import numpy as np 
import os
from itertools import chain
from scipy.fft import fft, fftfreq
from fooof import FOOOF
import seaborn as sns
import pandas as pd
import hashlib

def process_eeg_files(folder_path):
    data_frame = pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.endswith('.set'):
            file_path = os.path.join(folder_path, filename)
            
            # 1. Load Raw EEG Data
            raw_eeg = mne.io.read_raw_eeglab(file_path, preload= False)
            
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

            
            output_folder = "Z:/LeoF/Cygnus/Files"
            electrode_folder = "Z:/LeoF/Cygnus/Electrode"
            data_folder = "Z:/LeoF/Cygnus/DataFrame"
            for condition in conditions:

                # Extract data for condition
                epochs_data = epochs[condition].get_data(copy = True)
                
                # --- Compute pre/post masks & freq axis once per conditon ---
                
                times = epochs.times
                pre_mask = (times >= -0.6) & (times <= 0)
                post_mask = (times >= 0) & (times <= 0.6)
                
                n_samples_pre = pre_mask.sum()  # e.g., 154
                n_samples_post = post_mask.sum()
                
                sample_rate = epochs.info['sfreq']
                xf_pre = np.fft.fftfreq(n_samples_pre, 1 / sample_rate)
                xf_post = np.fft.fftfreq(n_samples_post, 1 / sample_rate)

                condition_array = np.zeros((len(electrodes),4,n_samples_pre)) # 3D array (28 electrodes × 4 metrics × 154 timepoints)

                for i_electrode, electrode in enumerate(electrodes):  
               
                    # Extract electrode data
                    electrode_data = epochs_data[:, i_electrode, :]  # Shape: (40, 309_times)
                    
                    # Split into pre/post stimulus
                    pre_data = electrode_data[:, pre_mask]  # Shape: (n_epochs, 154_pre_samples)
                    post_data = electrode_data[:, post_mask]  # Shape: (n_epochs, 154_post_samples)
                    
                    # Compute FFT in linear power
                    pre_spectra = np.abs(np.fft.fft(pre_data, axis=1)) ** 2  # Linear power
                    post_spectra = np.abs(np.fft.fft(post_data, axis=1)) ** 2

                    # Average across epochs
                    yf_avg_pre = pre_spectra.mean(axis=0)
                    yf_avg_post = post_spectra.mean(axis=0)

                    # ERP FFT (linear power)
                    erp = post_data.mean(axis=0)
                    fft_erp = np.abs(np.fft.fft(erp)) ** 2
        
                    # Post-minus-ERP 
                    post_minus_erp = yf_avg_post - fft_erp

                    n_samples_pre = pre_data.shape[1]
                    n_samples_post = post_data.shape[1]
                    
                    # --- Store in condition_array ---
                    condition_array[i_electrode, 0, :] = yf_avg_pre    # Metric 0: Pre-event power
                    condition_array[i_electrode, 1, :] = yf_avg_post   # Metric 1: Post-event power
                    condition_array[i_electrode, 2, :] = post_minus_erp # Metric 2: Post - ERP
                    condition_array[i_electrode, 3, :] = fft_erp # Metric 3: ERP
        
                    # FOOOF analysis
                    power_spec_windows = [
                        ("yf_avg_pre", yf_avg_pre, xf_pre),
                        ("yf_avg_post", yf_avg_post, xf_post),
                        ("post_minus_erp", post_minus_erp, xf_post)
                    ]
                    
                    for time_window, power_spec, xf in power_spec_windows:
                        # Fit FOOOF model
                        fg = FOOOF(
                            peak_width_limits=[2.5, 8],
                            max_n_peaks=1,
                            min_peak_height=0.3,
                            peak_threshold=2.0,
                            aperiodic_mode='fixed'
                        )
                        
                        try:
                            fg.fit(xf, power_spec, freq_range=(2, 25))
                            if fg.has_model:
                                exp = fg.get_params('aperiodic_params')[1]
                                offset = fg.get_params('aperiodic_params')[0]
                                rsq = fg.get_params('r_squared')
                            else:
                                exp, offset, rsq = np.nan, np.nan, np.nan
                        except Exception as e:
                            print(f"FOOOF failed for {electrode}, {condition}, {time_window}: {str(e)}")
                            exp, offset, rsq = np.nan, np.nan, np.nan
                        
                        # Append to DataFrame
                        new_row = pd.DataFrame({
                            'Participant': [filename.split('.')[0]],
                            'Electrode': [electrode],
                            'Condition': [condition],
                            'Time Window': [time_window],
                            'Exponent': [exp],
                            'Offset': [offset],
                            'Rsq': [rsq]
                        })
                        data_frame = pd.concat([data_frame, new_row], ignore_index=True)
                
                # Save condition power array
                output_path = os.path.join(output_folder, f"{filename}_{condition}_power.npy")
                np.save(output_path, condition_array)

                # Save frequency axes once per subject/condition
                freq_data = {
                    'xf_pre': xf_pre,
                    'xf_post': xf_post
                }
                output_freq_path = os.path.join(output_folder, f"{filename}_{condition}_freq.npy")
                np.save(output_freq_path, freq_data)  # Save as dictionary in .npy file

                # Saving out electrodes 
                output_path = os.path.join(electrode_folder, f"{filename}_{condition}_electrodes.npy")
                np.save(output_path, electrodes)

    # Saving out dataframe
    df_path = os.path.join(data_folder, "data_frame.csv")
    data_frame.to_csv(df_path)

    return data_frame

if __name__ == '__main__':
    folder_path = "C:/Users/leofl/OneDrive/Pictures/Documents/GitHub/EEG_ERP_Aperiodic_Analysis/files"
    y_frequencies_df = process_eeg_files(folder_path)

def validate_npy_data(folder_path):
    expected_shape = (28, 4, 154)  # Adjust if your dimensions differ
    hash_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            data = np.load(os.path.join(folder_path, filename))
            if data.shape != expected_shape:
                print(f"❌ {filename}: Shape {data.shape} (Expected {expected_shape})")
            else:
                print(f"✅ {filename}: Valid shape")
            
def violin_plots(path):
    """
    Function takes the dataframe a and slices it based on pip condition.
    With the sliced conditions, it uses takes the exponent values at 
    different time values to make seperate violin plots. This is done 
    through a for loop of the different sliced data frames.
    """
    a = pd.read_csv(path)

    # Flip exponent signs *before* subsetting
    a['exponent_negative'] = a['Exponent'] * -1

    # Filter by condition
    frequent_rare_df = a[a['Condition'] == 'frequent_rare'].groupby(by = ["Electrode", "Time Window"]).mean(numeric_only=True).reset_index()
    rare_frequent_df = a[a['Condition'] == 'rare_frequent'].groupby(by = ["Electrode", "Time Window"]).mean(numeric_only=True).reset_index()
    frequent_frequent_df = a[a['Condition'] == 'frequent_frequent'].groupby(by = ["Electrode", "Time Window"]).mean(numeric_only=True).reset_index()

    data = [frequent_rare_df, rare_frequent_df, frequent_frequent_df]
    titles = ["Frequent_Rare", "Rare_Frequent", "Frequent_Frequent"]
    hue_order = ['yf_avg_pre', 'yf_avg_post', 'post_minus_erp']
    

    for index, df in enumerate(data):
        plt.figure(figsize=(8, 5))
        # Fixed violinplot call
        ax = sns.violinplot(
            data=df,
            x='Time Window',
            y='exponent_negative',
            hue='Time Window',
            order = hue_order,
            hue_order = hue_order,  
            palette='pastel',
            inner='box',
            linewidth=1.1,
            cut = 0 
        )
        # Adjusted stripplot
        sns.stripplot(
            data=df,
            x='Time Window',
             y='exponent_negative',
            color='grey',
            alpha=0.4,  # More transparent
            jitter=0.08
        )

       
        plt.xlabel("Epoch Time Window")
        plt.ylabel("Exponent Value")
        plt.title(titles[index])
        plt.show()  # Remove ylim to auto-scale
violin_plots("Z:/LeoF/Cygnus/DataFrame/data_frame.csv")


