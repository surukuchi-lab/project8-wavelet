import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy as sp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import time
#import utils
import os
import json
from EggFileReader import EggFileReader
from scipy.signal import find_peaks, peak_widths

def map_index_to_x(x, a, out):
    returns = []
    for i in out:
        subreturn = []
        for j in i:
            idx = int(j)
            diff = j-idx
            subreturn.append(x[idx] + a*diff)
        returns.append(subreturn)

    returns = np.array(returns).T

    return returns

def recon_on_many(itr, ft=31e6, units=1e6, magnitude=False, rel=False, B=20, C=3):
    path = '/nfs/roberts/pi/pi_kmh66/spreadsheetData/SpreadsheetRow51_UNetScattering_CCA/Trap_CCA_long_trap_44_UNetScattering_1/Trap_CCA_long_trap_44_UNetScattering_1/results/'
    files = list(os.scandir(path))
    length = len(files)
    i = 0
    passer = 0
    error = np.zeros((2, itr))
    FWHMs = np.zeros((2, itr))

    while i < itr:
        filepath = path
        
        if not files[i].is_dir():
            i += 1
            itr += 1
            passer -= 1
        else:
            i += 1
    
            name = files[i].name
            filepath = filepath + name
    
            subfiles = os.scandir(filepath)
    
            for file in subfiles:
                if file.name.endswith('.egg'):
                    eggfilepath = filepath+"/"+file.name
    
            jsonfilepath = filepath+'/LocustEventProperties.json'
    
            with open(jsonfilepath, 'r') as file:
                metadata = json.load(file)
    
            file = EggFileReader(eggfilepath)
            stream = file.load_ts_stream()[0][0][0]
            I_data = stream.real
            Q_data = stream.imag
    
        
            if (i+passer-1) % 10 == 0: print(f"{i+passer}/{itr+passer}")
            file_id = files[i].name[2:-4]
            fs = float(metadata['run-parameters']['sampling-freq-mega-hz'])*units
            f = metadata['0']['0']['output-avg-frequency']
            fmix = f - ft
            #print(fmix)
            start = metadata['0']['0']['start-time']
            end = metadata['0']['0']['end-time']
            
            actual = ft/1e6
        
            idx = int(fs*start)
            edx = int(fs*end)
            
            data = I_data[idx:edx] + 1j*Q_data[idx:edx]
            T  = float(data.size)/fs
            N  = len(data)
            t  = np.arange(N) / fs
            signal_locust = data * (-1.0)**np.arange(N)
    
            signal = signal_locust * np.exp(-1j * 2*np.pi * fmix * t)
            
            t_center = 0.0015     # seconds -- time where there is a track
            window   = 5e-3       # 5 ms around it
            i0 = max(0, int((t_center - window/2) * fs))
            i1 = min(N, int((t_center + window/2) * fs))
            signal = signal[i0:i1]
            signal_locust = signal_locust[i0:i1]
            N = len(signal)
            t = np.arange(N) / fs
        
            ### STFT
            win_len = min(1024, N // 8)     
            hop     = max(1, win_len // 8)
            
            win = gaussian(win_len, std=win_len/6, sym=True)
            SFT = ShortTimeFFT(win, hop=hop, fs=fs, fft_mode='centered',
                               scale_to='magnitude')
            Sxx = SFT.stft(signal)
            f_stft = SFT.f
            t_stft = SFT.t(N)
    
            ### CWT
            wavelet = f'cmor{B}-{C}'
            
            fmin     = 50e5
            fmax     = 50e6
            n_voices = 96
             
            freqs_pos = np.logspace(np.log10(fmin), np.log10(fmax), n_voices)
            scales    = pywt.frequency2scale(wavelet, freqs_pos / fs)
    
            coefs, f_returned = pywt.cwt(
                signal, scales, wavelet, sampling_period=1/fs, method='fft'
            )
            mag = np.abs(coefs)
    

            means_CWT = np.mean(mag, axis=1)
            means_STFT = np.mean(np.abs(Sxx), axis=1)
            
            f_CWT = freqs_pos[np.argmax(means_CWT)]/1e6
            #print(f"\nFrequency CWT: {f_CWT}, \nDifference Error: {f_CWT-actual}\n")
            
            f_STFT = f_stft[np.argmax(means_STFT)]/1e6
            #print(f"Frequency STFT: {f_STFT}, \nDifference Error: {f_STFT-actual}")
                
            if rel: denom=actual
            else: denom=1
        
            if magnitude:
                error[0,i+passer-1]=np.abs((f_CWT-actual)/denom)
                error[1,i+passer-1]=np.abs((f_STFT-actual)/denom)
            else:
                error[0,i+passer-1] = (f_CWT-actual)/denom
                error[1,i+passer-1] = (f_STFT-actual)/denom

            m_CWT = np.sum(mag, axis=1)
            fullxrange = np.array(range(len(m_CWT)))        
            a_f = freqs_pos[1]-freqs_pos[0]
            n = m_CWT
            peaks = find_peaks(n)[0]
            
            #peaks = peaks[np.where(n[peaks] >= 50)[0]]
            widths = np.array(peak_widths(n, peaks, rel_height=0.5))
            
            f_bounds = map_index_to_x(freqs_pos[fullxrange], a_f, widths[2:])
            f_widths = np.zeros((len(f_bounds[:, 0]), 4))
            f_widths[:, 0] = f_bounds[:, 1] - f_bounds[:, 0]

            FWHMs[0,i+passer-1]=max(f_widths[:, 0])

            m_STFT = np.sum(np.abs(Sxx), axis=1)
            a_f = f_stft[1]-f_stft[0]
            n = m_CWT
            peaks = find_peaks(n)[0]
            #peaks = peaks[np.where(n[peaks] >= 100)[0]]
            widths = np.array(peak_widths(n, peaks, rel_height=0.5))
            
            f_bounds = map_index_to_x(f_stft[fullxrange], a_f, widths[2:])
            f_widths = np.zeros((len(f_bounds[:, 0]), 4))
            f_widths[:, 0] = f_bounds[:, 1] - f_bounds[:, 0]
            
            FWHMs[1,i+passer-1]=max(f_widths[:, 0])
            

            
    print("COMPLETE")    
    np.save(f"error_{itr+passer}_mag{magnitude}_B{B}_C{C}", error)
    np.save(f"FWHMs_{itr+passer}_mag{magnitude}_B{B}_C{C}", FWHMs)

recon_on_many(100)