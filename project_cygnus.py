"""
Leo Fleury
Codename: Project Cygnus 

Gryurkovics does an anlaysis using a Visual & Audio Oddball tasks which tend 
to illicit the P300 and the MMN (mismatch negativity). The first part of this
reanalysis will be to apply the same processing steps as Gyrukovics entirely
in Python/MNE using data from ERP Core to see if the results can be replicated.
I'm tryinh to apply it to P3 and MMN ERPs. 
"""

import mne
import matplotlib.pyplot as plt
import numpy as np 
from scipy.fft import fft, fftfreq

# Read the file using MNE
file_path = "C:/Users/leofl/OneDrive/Desktop/ERP Data/Project Cygnus/p3_preprocessed.set"
p3 = mne.io.read_raw_eeglab(file_path, preload=True)

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
p3.rename_channels({"FP1":'Fp1','FP2':'Fp2'})
p3.set_channel_types({'HEOG_left':'eog', 'HEOG_right':'eog', \
                       'VEOG_lower':'eog', '(uncorr) HEOG' : 'eog',\
                       '(uncorr) VEOG' : 'eog'})
p3.set_montage(ten_twenty_montage)

#filtering and scaling bc other formats will make the data look unreasonable
p3.copy().filter(0.1, 100).plot(scalings=0.00008, clipping=None)
# plt.show() 

event_p3, event_id_p3 = mne.events_from_annotations(p3)

#removing wrong or right markers in event p3 (in this case 6 or 7)
event_p3_minus67 = [array for array in event_p3 if array[2] \
                    != 6 and array[2] != 7]


event_id_p3_epoch = {
    "target": [1, 9, 15, 21, 27],  # got the #s fom the event_id array (11,22 etc. were the oddball events/ triggers)
    "nontarget": [2, 3, 4, 5, 8, 10, 11, 12, 13,
                  14, 16, 17, 18, 19,
                  20, 22, 23, 24, 25, 
                  26]
}


for array in range(len(event_p3_minus67)):
    if event_p3_minus67[array][2] in event_id_p3_epoch['target']:
        event_p3_minus67[array][2] = 100
    elif array > 0 and event_p3_minus67[array][2] in event_id_p3_epoch['nontarget'] and event_p3_minus67[array - 1][2] == 100:
        event_p3_minus67[array][2] = 200
    elif array > 0 and event_p3_minus67[array][2] in event_id_p3_epoch['nontarget'] and event_p3_minus67[array - 1][2] == 200 or event_p3_minus67[array - 1][2] == 300:
        event_p3_minus67[array][2] = 300


new_event_id_p3_epoch = {
      "frequent_rare": 100, 
      "rare_frequent": 200, 
      "frequent_frequent": 300
}


epochs = mne.Epochs(p3, events = event_p3_minus67, \
                    event_id = new_event_id_p3_epoch,\
                    tmin  = -1, tmax = 1\
                    ,baseline = None,
                    preload=True) 


# fft and pre and post stimulus windows
epochs_pre_stimulus_window = epochs.copy().crop(tmin=-1, tmax=0)
epochs_post_stimulus_window = epochs.copy().crop(tmin=0, tmax= 1)

# computing ERP and Mean Pre and Post 
mean_post_window =  epochs_pre_stimulus_window.average()
mean_pre_window = epochs_post_stimulus_window.average()
erp = epochs.average() # look at the averages for target, non target, and target - nontarget (and compare all 3 of them )




#preparing for the fft
sample_rate = epochs.info['sfreq']
duration = len(epochs.times) / sample_rate
N = sample_rate * duration
post_data = epochs_post_stimulus_window.get_data()

yf = fft(post_data)
xf = fftfreq(N, 1 / sample_rate)

plt.plot(xf, np.abs(yf))
plt.show()

#calculating post_minus ERP

    
