import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os
import sys
import re
#from tqdm import tqdm
#import h5py as h5
import numpy as np
from EggFileReader import EggFileReader
#from ssqueezepy import Wavelet, cwt, extract_ridges


def retrieval(itr):
    path= '/nfs/roberts/pi/pi_kmh66/spreadsheetData/SpreadsheetRow51_UNetScattering_CCA/Trap_CCA_long_trap_44_UNetScattering_1/Trap_CCA_long_trap_44_UNetScattering_1/results/'
    files = list(os.scandir(path))
    length = len(files)
    itr = itr
    count = 0
    
    while count < itr:
        filepath = path
        
        if not files[count].is_dir():
            continue
        else:
            count += 1
    
            name = files[count].name
            filepath = filepath + name
    
            subfiles = os.scandir(filepath)
    
            for file in subfiles:
                if file.name.endswith('.egg'):
                    eggfilepath = filepath+"/"+file.name
    
            jsonfilepath = filepath+'/LocustEventProperties.json'
    
            with open(jsonfilepath, 'r') as file:
                data = json.load(file)
    
            file = EggFileReader(eggfilepath)
            stream = file.load_ts_stream()[0][0][0]
            np.save(f"data/I_{name}", stream.real)
            np.save(f"data/Q_{name}", stream.imag)
            with open(f"data/metadata_{name}.json", 'w') as file:
                json.dump(data, file)

def get_quantities(path):
    with open(path) as file:
        metadata = json.load(file)
        
    fs = metadata['run-parameters']['sampling-freq-mega-hz']
    n = metadata['0']['ntracks']
    f = metadata['0']['0']['output-avg-frequency']
    times = np.zeros((2, n))

    for i in range(n):
        times[0, i] = metadata['0'][f"{i}"]['start-time']
        times[1, i] = metadata['0'][f"{i}"]['end-time']

    return fs, times, f
        

def visual(Wx, wavelet, w=10, units=1e6, fs=403e6, numticks=12, save=False):
    Wxr = np.abs(Wx) # make real

    idxs = []
    for i in range(len(Wxr[0, :])): # find approximate location of ridge
        Wxri = Wxr[:, i]
        idxs.append(np.argmax(Wxri))
    
    centerW = int(np.mean(idxs))
    stdW = int(np.std(idxs)) # cropping parameters
    if stdW==0: stdW+=1
    
    def fmt(ticks): # for formatting
        if all(isinstance(h, str) for h in ticks):
            return "%s"
        return ("%.d" if all(float(h).is_integer() for h in ticks) else "%.2f")

    idxs = np.argmax(wavelet, axis=-1)
    ticks = idxs/(len(wavelet[1])) * fs / units
    
    lowlim=centerW-w*stdW # cropping bounds
    uplim=centerW+w*stdW
    
    ticks_crop = ticks[lowlim:uplim] # crop
    Wxr_crop = Wxr[lowlim:uplim, :]
    
    ii = np.linspace(0, len(ticks_crop) - 1, num_ticks).astype('int32') # set yaxis ticks and labels
    yt = np.array([fmt(ticks_crop) % h for h in ticks_crop[ii]])

    fig, ax = plt.subplots() # plot
    ax.set_yticks(ii)
    ax.set_yticklabels(yt)
    
    image = ax.imshow(Wxr_crop, aspect='auto')

    plt.ylabel("Frequency (MHz)")
    plt.xlabel("Index")
    fig.colorbar(image, label="Amplitude")
    plt.show()

def ridges_on_many(itr, path='data', units=1e6, magnitude=False, rel=False):
    obj = list(os.scandir(path))
    if itr >= len(obj)//3 or itr <= 0:
        itr = len(obj)//3
    
    units = 1e6 # except for ridge apparently
    
    error = np.zeros(itr-1)
    
    for i in range(1, itr): # to ignore python notebook
        print(f"{i}/{itr}")
        file_id = obj[i].name[2:-4]
        metadata = get_quantities('data/metadata_'+file_id+'.json')
        fs = float(metadata[0]) * units
        start = metadata[1][0, 0]
        end = metadata[1][1, 0]
        I_data = np.load('data/'+obj[i].name)
        Q_data = np.load('data/Q_'+obj[i].name[2:])
    
        idx = int(fs*start)
        edx = int(fs*end)
        data = np.sqrt(I_data[idx:edx]**2+Q_data[idx:edx]**2)
    
        wavelet = Wavelet(('morlet', {'mu': 5}))
        Wx, scales = cwt(data, wavelet, fs=fs, nv=32)
        psis = wavelet(scale=scales, N=len(I_data))
    
        ridge_idxs = extract_ridges(Wx, scales, penalty=20, n_ridges=1, bw=25)
    
        idxs = np.argmax(psis, axis=-1)
        ticks = idxs/(len(psis[1])) * fs / units
        
        ridge = np.mean(ticks[ridge_idxs])*units*10 # for error
        actual = metadata[2]

        if rel: denom=actual
        else: denom=1
    
        if magnitude: error[i-1]=np.abs((ridge-actual)/denom)
        else: error[i-1] = (ridge-actual)/denom
        
    print("COMPLETE")    
    np.save(f"error_{itr}_CWT_mag{magnitude}", error)

