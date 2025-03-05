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

            
            output_folder = "Z:/LeoF/Cygnus/Files"
            electrode_folder = "Z:/LeoF/Cygnus/Electrode"
            data_folder = "Z:/LeoF/Cygnus/DataFrame"
            for condition in conditions:
                condition_array = np.zeros((28,4,154)) # 3D array (28 electrodes × 4 metrics × 154 timepoints)
                for i_electrode, electrode in enumerate(electrodes):  
                    # Get data for this condition and electrode
                    epochs_data = epochs[condition].get_data(copy=True)  # Shape: (n_epochs, 28_channels, 309_times)
                    times = epochs.times
                    
                    # Extract electrode data
                    electrode_idx = epochs.ch_names.index(electrode)
                    electrode_data = epochs_data[:, electrode_idx, :]  # Shape: (n_epochs, 309_times)
                    
                    # Split into pre/post stimulus
                    pre_mask = (times >= -0.6) & (times <= 0)
                    post_mask = (times >= 0) & (times <= 0.6)
                    pre_data = electrode_data[:, pre_mask]  # Shape: (n_epochs, 154_pre_samples)
                    post_data = electrode_data[:, post_mask]  # Shape: (n_epochs, 154_post_samples)
                    
                    # Compute power spectra (dB) and trim to positive frequencies
                    n_samples_pre = pre_data.shape[1]
                    n_samples_post = post_data.shape[1]
                    
                    # Compute FFT in linear power
                    pre_spectra = np.abs(np.fft.fft(pre_data, axis=1)) ** 2  # Linear power
                    post_spectra = np.abs(np.fft.fft(post_data, axis=1)) ** 2

                    # Trim to positive frequencies
                    n_samples_pre = pre_data.shape[1]
                    pre_spectra = pre_spectra[:, :n_samples_pre // 2]
                    yf_avg_pre = pre_spectra.mean(axis=0)

                    n_samples_post = post_data.shape[1]
                    post_spectra = post_spectra[:, :n_samples_post // 2]
                    yf_avg_post = post_spectra.mean(axis=0)

                    # ERP FFT (linear power)
                    erp = post_data.mean(axis=0)
                    fft_erp = np.abs(np.fft.fft(erp)) ** 2
                    fft_erp = fft_erp[:n_samples_post // 2]
                    
                    # Post-minus-ERP (ensure matching lengths)
                    min_len = min(len(yf_avg_post), len(fft_erp))
                    post_minus_erp = yf_avg_post[:min_len] - fft_erp[:min_len]
                    
                    # Get frequency axes
                    sample_rate = epochs.info['sfreq']
                    xf_pre = np.fft.fftfreq(n_samples_pre, 1 / sample_rate)[:n_samples_pre // 2]
                    xf_post = np.fft.fftfreq(n_samples_post, 1 / sample_rate)[:n_samples_post // 2]
                    
                    # FOOOF analysis
                    power_spec_windows = [
                        ("yf_avg_pre", yf_avg_pre, xf_pre),
                        ("yf_avg_post", yf_avg_post, xf_post),
                        ("post_minus_erp", post_minus_erp, xf_post)
                    ]
                    
                    for time_window, power_spec, xf in power_spec_windows:
                        # Skip invalid spectra
                        if np.isnan(power_spec).any() or np.all(power_spec == 0):
                            exp, offset, rsq = np.nan, np.nan, np.nan
                            continue
                            
                        # Fit FOOOF model
                        fg = FOOOF(
                            peak_width_limits=[2.0, 12.0],  # Wider search range
                            max_n_peaks=2, 
                            min_peak_height=0.1,  # Lower threshold for dB
                            peak_threshold=1.5,
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
                # Save per-condition data
                output_path = os.path.join(output_folder, f"{filename}_{condition}_power.npy")
                np.save(output_path, condition_array)

                # Saving out electrodes -- might need to do by participant if electrodes where removed
                output_path = os.path.join(electrode_folder, f"{filename}_{condition}_electrodes.npy")
                np.save(output_path, electrodes)

    # Saving out dataframe
    df_path = os.path.join(data_folder, "data_frame.csv")
    data_frame.to_csv(df_path)

    return data_frame

if __name__ == '__main__':
    folder_path = "C:/Users/leofl/OneDrive/Pictures/Documents/GitHub/Project-Cygnus/files"
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
            
            # 2. Check for duplicate values (using hashes)
            data_bytes = data.tobytes()  # Convert array to bytes
            data_hash = hashlib.sha256(data_bytes).hexdigest()  # Generate unique hash
            
            if data_hash in hash_dict:
                print(f"❌ DUPLICATE DATA: {filename} matches {hash_dict[data_hash]}")
            else:
                hash_dict[data_hash] = filename
                print(f"✅ {filename}: Not duplicate data")

def validate_df_shape(path):
    df = pd.read_csv(path)
    df_shape = df.shape
    if df_shape != (5040, 8): # 20 × 3 × 3 × 28 = 5,040 rows
        print(f"❌ data_frame: Shape {df_shape} (Expected {df_shape})")
    else: 
        print(f"✅ data_frame: Valid shape")

def violin_plots(path):
    """
    Function takes the dataframe a and slices it based on pip condition.
    With the sliced conditions, it uses takes the exponent values at 
    different time values to make seperate violin plots. This is done 
    through a for loop of the different sliced data frames.
    """
    a = pd.read_csv(path)
    print("Exponent stats:\n", a['Exponent'].describe())  # Debug data

    # Filter by condition
    frequent_rare_df = a[a['Condition'] == 'frequent_rare']
    rare_frequent_df = a[a['Condition'] == 'rare_frequent']
    frequent_frequent_df = a[a['Condition'] == 'frequent_frequent']

    data = [frequent_rare_df, rare_frequent_df, frequent_frequent_df]
    titles = ["Frequent_Rare", "Rare_Frequent", "Frequent_Frequent"]

    for index, df in enumerate(data):
        plt.figure(figsize=(8, 5))
        # Fixed violinplot call
        sns.violinplot(
            data=df,
            x='Time Window',
            y='Exponent',
            hue='Time Window',  
            palette='pastel',
            inner='box',
            linewidth=1.1,
            legend=False
        )
        # Adjusted stripplot
        sns.stripplot(
            data=df,
            x='Time Window',
            y='Exponent',
            color='grey',
            alpha=0.4,  # More transparent
            jitter=0.08
        )
        plt.xlabel("Epoch Time Window")
        plt.ylabel("Exponent Value")
        plt.title(titles[index])
        plt.show()  # Remove ylim to auto-scale

validate_npy_data("Z:/LeoF/Cygnus/Files")
validate_df_shape("Z:/LeoF/Cygnus/DataFrame/data_frame.csv")
violin_plots("Z:/LeoF/Cygnus/DataFrame/data_frame.csv")

