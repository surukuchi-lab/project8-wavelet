import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy as sp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian, boxcar
import time
#import utils
import os
import json
from EggFileReader import EggFileReader
from scipy.signal import find_peaks, peak_widths
from extract_lfa_b_field import *


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
    #path = '/nfs/roberts/pi/pi_kmh66/spreadsheetData/SpreadsheetRow51_UNetScattering_CCA/Trap_CCA_long_trap_44_UNetScattering_1/Trap_CCA_long_trap_44_UNetScattering_1/results/'
    path = '/nfs/roberts/scratch/pi_kmh66/vs553/Wavelet_data/results/'
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
            """
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
            n = m_STFT
            peaks = find_peaks(n)[0]
            #peaks = peaks[np.where(n[peaks] >= 100)[0]]
            widths = np.array(peak_widths(n, peaks, rel_height=0.5))
            
            f_bounds = map_index_to_x(f_stft[fullxrange], a_f, widths[2:])
            f_widths = np.zeros((len(f_bounds[:, 0]), 4))
            f_widths[:, 0] = f_bounds[:, 1] - f_bounds[:, 0]
            
            FWHMs[1,i+passer-1]=max(f_widths[:, 0])
            """
            

            
    print("COMPLETE")    
    np.save(f"error_{itr+passer}_mag{magnitude}_B{B}_C{C}", error)
    np.save(f"FWHMs_{itr+passer}_mag{magnitude}_B{B}_C{C}", FWHMs)

def energy_SI(f, B):
    m_e = 9.1093837139e-31
    e = 1.60217733e-19
    c = 299792458
    return (e*B/(2*np.pi * f) - m_e)*c**2 * 6.241509e18 # J -> eV conversion

def energy_on_many(itr, path, ft=31e6, units=1e6, magnitude=False, rel=False, B=20, C=3, num_batches=5, LOFREQ=25970200000.0):
    files = [i for i in os.scandir(path)]
    length = len(files)
    if itr >= length:
        itr = length-1
    i = 0
    error = np.zeros((2, itr))
    error_fft = np.zeros((2, itr))
    error_energy = np.zeros((2, itr))
    FWHMs = np.zeros((2, itr))

    while i < itr:
        filepath = path
        
        i += 1
    
        name = files[i].name
        filepath = filepath+name

        subfiles = os.scandir(filepath)

        for file in subfiles:
            if file.name.endswith('.egg'):
                eggfilepath = filepath+"/"+file.name

            if file.name.endswith('.root') and file.name.startswith('KassCavitySeed'):
                bpath = filepath+"/"+file.name

        jsonfilepath = filepath+'/LocustEventProperties.json'

        with open(jsonfilepath, 'r') as file:
            metadata = json.load(file)

        file = EggFileReader(eggfilepath)
        stream = file.load_ts_stream()[0][0][0]
        I_data = stream.real
        Q_data = stream.imag

    
        if (i-1) % 10 == 0: print(f"{i}/{itr}")
        fs = float(metadata['run-parameters']['sampling-freq-mega-hz'])*units
        f = metadata['0']['0']['output-avg-frequency']
        fmix = f - ft
        start = metadata['0']['0']['start-time']
        end = metadata['0']['0']['end-time']

        ENERGY = metadata['0']['0']['energy-ev']
        
        actual = ft/1e6
    
        idx = int(fs*start)
        edx = int(fs*end)

        rng = np.linspace(idx, edx, num_batches, endpoint=True)
        rng = [int(j) for j in rng]

        fcs = []

        ### STFT

        data = I_data[idx:edx] + 1j*Q_data[idx:edx]
        T  = float(data.size)/fs
        N  = len(data)
        t  = np.arange(N) / fs
        signal_locust = data * (-1.0)**np.arange(N)
        signal = signal_locust * np.exp(-1j * 2*np.pi * fmix * t)

        """
        t_center = (end-start)/2     # seconds -- time where there is a track
        window   = 5e-3       # 5 ms around it
        i0 = max(0, int((t_center - window/2) * fs))
        i1 = min(N, int((t_center + window/2) * fs))
        signal = signal[i0:i1]
        signal_locust = signal_locust[i0:i1]
        """
        N = len(signal)
        t = np.arange(N) / fs
        
        win_len = min(1024, N // 8)     
        hop     = max(1, win_len // 8)
        
        win = boxcar(win_len, sym=True)
        SFT = ShortTimeFFT(win, hop=hop, fs=fs, fft_mode='centered',
                           scale_to='magnitude')
        Sxx = SFT.stft(signal)
        f_stft = SFT.f
        t_stft = SFT.t(N)

        means_STFT = np.mean(np.abs(Sxx), axis=1)
        inits_STFT = Sxx[:, 0]
        
        freq_cands = np.where(np.abs(f_stft - ft) <= 5e6)[0]
        f_STFT = 0
        
        if freq_cands.size > 0:
            cands = means_STFT[freq_cands]
            f_STFT = f_stft[freq_cands[np.argmax(cands)]]/1e6
            
            init_cands = inits_STFT[freq_cands]
            init_f_STFT = f_stft[freq_cands[np.argmax(cands)]]/1e6
        else:
            f_STFT = f_stft[np.argmax(means_STFT)]/1e6

            init_f_STFT = f_stft[np.argmax(inits_STFT)]/1e6


        ### CWT
        init_f_CWT = 0
        for j in range(len(rng)-1):
            data = I_data[rng[j]:rng[j+1]] + 1j*Q_data[rng[j]:rng[j+1]]
            T  = float(data.size)/fs
            N  = len(data)
            t  = np.arange(N) / fs
            signal_locust = data * (-1.0)**np.arange(N)
    
            signal = signal_locust * np.exp(-1j * 2*np.pi * fmix * t)
    
            """
            t_center = (end-start)/2     # seconds -- time where there is a track
            window   = 5e-3       # 5 ms around it
            i0 = max(0, int((t_center - window/2) * fs))
            i1 = min(N, int((t_center + window/2) * fs))
            signal = signal[i0:i1]
            signal_locust = signal_locust[i0:i1]
            """
            N = len(signal)
            t = np.arange(N) / fs
        
            
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
    
            freq_cands = np.where(np.abs(freqs_pos - ft) <= 5e6)[0]
                    
            if freq_cands.size > 0:
                cands = means_CWT[freq_cands]
                f_CWT = freqs_pos[freq_cands[np.argmax(cands)]]/1e6
                fcs.append(f_CWT)

                if j == 0:
                    inits_CWT = mag[:, 0]
                    init_cands = inits_CWT[freq_cands]
                    init_f_CWT = freqs_pos[freq_cands[np.argmax(cands)]]/1e6
            else:
                f_CWT = freqs_pos[np.argmax(means_CWT)]/1e6
                fcs.append(f_CWT)

                if j == 0:
                    inits_CWT = mag[:, 0]
                    init_f_CWT = freqs_pos[np.argmax(inits_CWT)]/1e6

        ### FFT
        data = I_data[idx:edx] + 1j*Q_data[idx:edx]
        T  = float(data.size)/fs
        N  = len(data)
        t  = np.arange(N) / fs
        signal_locust = data * (-1.0)**np.arange(N)
        signal = signal_locust * np.exp(-1j * 2*np.pi * fmix * t)
        
        fft_vals = np.fft.fft(signal)
        fft_freqs = np.fft.fftfreq(N, d=1/fs)

        
        freq_cands = np.where(np.abs(fft_freqs - ft) <= 5e6)[0]
        f_FFT = 0
        
        if freq_cands.size > 0:
            cands = fft_vals[freq_cands]
            f_FFT = fft_freqs[freq_cands[np.argmax(cands)]]/1e6
        else:
            f_FFT = fft_freqs[np.argmax(fft_vals)]/1e6


        ### ENERGY

        
        b_avg, n_total, t_start_total, t_end_total = time_weighted_average(bpath, t_min=0, t_max=1/fs)
        CWT_energy = energy_SI(init_f_CWT*1e6 + fmix + LOFREQ, b_avg)
        STFT_energy = energy_SI(init_f_STFT*1e6 + fmix + LOFREQ, b_avg)


        if rel: denom=actual
        else: denom=1
    
        if magnitude:
            error[0,i-1]=np.abs((np.mean(fcs)-actual)/denom)
            error_fft[0, i-1] = np.abs((np.mean(fcs)-f_FFT)/denom)
            error[1,i-1]=np.abs((f_STFT-actual)/denom)
            error_fft[1, i-1] = np.abs((f_STFT-f_FFT)/denom)

            error_energy[0, i-1] = np.abs(CWT_energy - ENERGY)
            error_energy[1, i-1] = np.abs(STFT_energy - ENERGY)
        else:
            error[0,i-1] = (np.mean(fcs)-actual)/denom
            error_fft[0, i-1] = (np.mean(fcs)-f_FFT)/denom
            error[1,i-1] = (f_STFT-actual)/denom
            error_fft[1, i-1] = (f_STFT-f_FFT)/denom

            error_energy[0, i-1] = CWT_energy - ENERGY
            error_energy[1, i-1] = STFT_energy - ENERGY

        ### FWHMs

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

        FWHMs[0,i-1]=max(f_widths[:, 0])

        m_STFT = np.sum(np.abs(Sxx), axis=1)
        a_f = f_stft[1]-f_stft[0]
        n = m_STFT
        peaks = find_peaks(n)[0]
        #peaks = peaks[np.where(n[peaks] >= 100)[0]]
        widths = np.array(peak_widths(n, peaks, rel_height=0.5))

        fullxrange = np.array(range(len(f_stft)))
        a_f = f_stft[1]-f_stft[0]
        f_bounds = map_index_to_x(f_stft[fullxrange], a_f, widths[2:])
        f_widths = np.zeros((len(f_bounds[:, 0]), 4))
        f_widths[:, 0] = f_bounds[:, 1] - f_bounds[:, 0]
        
        FWHMs[1,i-1]=max(f_widths[:, 0])


    print("COMPLETE")    
    np.save(f"error_{itr}_mag{magnitude}_B{B}_C{C}", error)
    np.save(f"error_FFT_{itr}_mag{magnitude}_B{B}_C{C}", error_fft)
    np.save(f"error_ENERGY_{itr}_mag{magnitude}_B{B}_C{C}", error_energy)
    np.save(f"FWHMs_{itr}_mag{magnitude}_B{B}_C{C}", FWHMs)

        

def batch_on_many(itr, ft=31e6, units=1e6, magnitude=False, rel=False, B=20, C=3, num_batches=5):
    path = '/nfs/roberts/pi/pi_kmh66/spreadsheetData/SpreadsheetRow51_UNetScattering_CCA/Trap_CCA_long_trap_44_UNetScattering_1/Trap_CCA_long_trap_44_UNetScattering_1/results/'
    files = [i[0] for i in os.walk(path)]
    length = len(files)
    if itr >= length:
        itr = length-1
    i = 0
    error = np.zeros((2, itr))
    error_fft = np.zeros((2, itr))
    FWHMs = np.zeros((2, itr))

    while i < itr:
        filepath = path
        
        i += 1
    
        name = files[i]
        filepath = name

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

    
        if (i-1) % 10 == 0: print(f"{i}/{itr}")
        fs = float(metadata['run-parameters']['sampling-freq-mega-hz'])*units
        f = metadata['0']['0']['output-avg-frequency']
        fmix = f - ft
        #print(fmix)
        start = metadata['0']['0']['start-time']
        end = metadata['0']['0']['end-time']
        
        actual = ft/1e6
    
        idx = int(fs*start)
        edx = int(fs*end)

        rng = np.linspace(idx, edx, num_batches, endpoint=True)
        rng = [int(j) for j in rng]

        fcs = []
        fss = []

        for j in range(len(rng)-1):
            data = I_data[rng[j]:rng[j+1]] + 1j*Q_data[rng[j]:rng[j+1]]
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

            freq_cands = np.where(np.abs(freqs_pos - ft) <= 5e6)[0]
                    
            if freq_cands.size > 0:
                cands = means_CWT[freq_cands]
                f_CWT = freqs_pos[freq_cands[np.argmax(cands)]]/1e6
                fcs.append(f_CWT)
            else:
                f_CWT = freqs_pos[np.argmax(means_CWT)]/1e6
                fcs.append(f_CWT)


        ### STFT

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
        
        win_len = min(1024, N // 8)     
        hop     = max(1, win_len // 8)
        
        win = gaussian(win_len, std=win_len/6, sym=True)
        SFT = ShortTimeFFT(win, hop=hop, fs=fs, fft_mode='centered',
                           scale_to='magnitude')
        Sxx = SFT.stft(signal)
        f_stft = SFT.f
        t_stft = SFT.t(N)

        means_STFT = np.mean(np.abs(Sxx), axis=1)
        
        freq_cands = np.where(np.abs(f_stft - ft) <= 5e6)[0]
        f_STFT = 0
        
        if freq_cands.size > 0:
            cands = means_STFT[freq_cands]
            f_STFT = f_stft[freq_cands[np.argmax(cands)]]/1e6
        else:
            f_STFT = f_stft[np.argmax(means_STFT)]/1e6

        ### FFT
        data = I_data[idx:edx] + 1j*Q_data[idx:edx]
        T  = float(data.size)/fs
        N  = len(data)
        t  = np.arange(N) / fs
        signal_locust = data * (-1.0)**np.arange(N)
        signal = signal_locust * np.exp(-1j * 2*np.pi * fmix * t)
        
        fft_vals = np.fft.fft(signal)
        fft_freqs = np.fft.fftfreq(N, d=1/fs)

        
        freq_cands = np.where(np.abs(fft_freqs - ft) <= 5e6)[0]
        f_FFT = 0
        
        if freq_cands.size > 0:
            cands = fft_vals[freq_cands]
            f_FFT = fft_freqs[freq_cands[np.argmax(cands)]]/1e6
        else:
            f_FFT = fft_freqs[np.argmax(fft_vals)]/1e6

            
        if rel: denom=actual
        else: denom=1
    
        if magnitude:
            error[0,i-1]=np.abs((np.mean(fcs)-actual)/denom)
            error_fft[0, i-1] = np.abs((np.mean(fcs)-f_FFT)/denom)
            error[1,i-1]=np.abs((f_STFT-actual)/denom)
            error_fft[1, i-1] = np.abs((f_STFT-f_FFT)/denom)
        else:
            error[0,i-1] = (np.mean(fcs)-actual)/denom
            error_fft[0, i-1] = (np.mean(fcs)-f_FFT)/denom
            error[1,i-1] = (f_STFT-actual)/denom
            error_fft[1, i-1] = (f_STFT-f_FFT)/denom
        """
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

        fullxrange = np.array(range(len(f_stft)))
        a_f = f_stft[1]-f_stft[0]
        f_bounds = map_index_to_x(f_stft[fullxrange], a_f, widths[2:])
        f_widths = np.zeros((len(f_bounds[:, 0]), 4))
        f_widths[:, 0] = f_bounds[:, 1] - f_bounds[:, 0]
        
        FWHMs[1,i+passer-1]=max(f_widths[:, 0])
        """
        

            
    print("COMPLETE")    
    np.save(f"error_{itr}_mag{magnitude}_B{B}_C{C}", error)
    np.save(f"error_FFT_{itr}_mag{magnitude}_B{B}_C{C}", error_fft)
    np.save(f"FWHMs_{itr}_mag{magnitude}_B{B}_C{C}", FWHMs)

"""
batch_on_many(600, B=20, C=3, ft=31e6)
"""

PATH = '/nfs/roberts/pi/pi_kmh66/spreadsheetData/SpreadsheetRow62_waveletData/Trap_CCA_long_trap_44_UNetScattering_1/results/'
energy_on_many(550, PATH, B=20, C=3, ft=31e6)