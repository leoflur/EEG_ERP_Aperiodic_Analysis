"""
Leo Fleury
Codename: Project Cygnus 

Gryurkovics does an anlaysis using a Visual & Audio Oddball tasks which tend 
to illicit the P300 and the MMN (mismatch negativity). The first part of this
reanalysis will be to apply the same processing steps as Gyrukovics entirely
in Python/MNE using data from ERP Core to see if the results can be replicated.
"""

import mne
import matplotlib.pyplot as plt

p3 = mne.io.read_raw_eeglab("C:\\Users\\leofl\\OneDrive\\Desktop\\ERP Data\\Cygnus\\data\\p3_preprocessed.set", preload = True)
mmn = mne.io.read_raw_eeglab("C:\\Users\\leofl\\OneDrive\\Desktop\\ERP Data\\Cygnus\\data\\mmn_preprocessed.set", preload = True)

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
p3.rename_channels({"FP1":'Fp1','FP2':'Fp2'})
mmn.rename_channels({"FP1":'Fp1','FP2':'Fp2'})
p3.set_channel_types({'HEOG_left':'eog', 'HEOG_right':'eog', \
                      'VEOG_lower':'eog', '(uncorr) VEOG' : 'eog', '(uncorr) HEOG': 'eog'})
mmn.set_channel_types({'HEOG_left':'eog', 'HEOG_right':'eog', \
                      'VEOG_lower':'eog', '(uncorr) VEOG' : 'eog', '(uncorr) HEOG': 'eog'})
p3.set_montage(ten_twenty_montage)
mmn.set_montage(ten_twenty_montage)

#filtering and scaling & saving to figure vars & keeping them open w. plt.show
# p3_fig = p3.copy().filter(0.1, 100).plot(scalings=0.00008, clipping=None)
# mmn_fig = mmn.copy().filter(0.1, 100).plot(scalings=0.00008, clipping=None)
# plt.show() 

event_p3, event_id_p3 = mne.events_from_annotations(p3)
event_mmn, event_id_mmn = mne.events_from_annotations(p3)
print(event_p3)
