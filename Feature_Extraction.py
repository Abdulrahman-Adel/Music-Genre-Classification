# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:04:33 2020

@author: Abdelrahman
"""

import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import os
import csv

main_dir = "Data\genres_original"

header = "filename chroma_stft_mean chroma_stft_var chroma_stft_std rmse_mean rmse_var rmse_std spectral_centroid_mean spectral_centroid_var spectral_centroid_std " 
header +=  "spectral_bandwidth_mean spectral_bandwidth_var spectral_bandwidth_std rolloff_mean rolloff_var rolloff_std zero_crossing_rate_mean zero_crossing_rate_var zero_crossing_rate_std"
for i in range(1, 21):
    header += f' mfcc_mean{i}'
for i in range(1, 21):
    header += f' mfcc_var{i}'
for i in range(1, 21):
    header += f' mfcc_std{i}'    
header += ' label'
header = header.split()

file = open('features.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for filename in os.listdir(main_dir):
    genre_path = os.path.join(main_dir,filename)
    for audio_name in os.listdir(genre_path):
        label = audio_name.split(".")[0]
        try:
            signal, sr = librosa.load(os.path.join(genre_path,audio_name), mono=True, duration=30)
        except:
            continue
        rmse = librosa.feature.rms(y=signal)[0]
        chroma_stft = librosa.feature.chroma_stft(y=signal, sr=sr)[0]
        spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(signal)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr)
        to_append = f"{audio_name} {np.mean(chroma_stft)} {np.var(chroma_stft)} {np.std(chroma_stft)} {np.mean(rmse)} {np.var(rmse)} {np.std(rmse)} " 
        to_append += f"{np.mean(spec_cent)} {np.var(spec_cent)} {np.std(spec_cent)} {np.mean(spec_bw)} {np.var(spec_bw)} {np.std(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.std(rolloff)} {np.mean(zcr)} {np.var(zcr)} {np.std(zcr)}"
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        for e in mfcc:
            to_append += f' {np.var(e)}'
        for e in mfcc:
            to_append += f' {np.std(e)}'    
        to_append += f' {label}'
        file = open('features.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
        


