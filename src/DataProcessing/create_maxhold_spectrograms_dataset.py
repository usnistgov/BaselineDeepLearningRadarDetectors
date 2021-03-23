
# NIST-developed software is provided by NIST as a public service. 
# You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. 
# You may improve, modify and create derivative works of the software or any portion of the software,
# and you may copy and distribute such modifications or works. 
# Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. 
# Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." 
# NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, 
# THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. 
# NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, 
# OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, 
# INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, 
# including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, 
# and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. 
# The software developed by NIST employees is not subject to copyright protection within the United States.

#%%
import numpy as np
import h5py
from pathlib import Path
import pandas as pd
from scipy import signal
#%%
RFDatasetDir = '../Dataset/RFDataset'
RFDatasetDir=r'\\fs1.ctl.nist.gov\www\RFDataset\FourGroupsSet_SNR(10-20)_NPow(-109dBmPerMHz)_duration(80ms)_FreqShiftRandomTimeVar'

#Generate training and validation file names
fileNames=[]
waveform_vars=[]
status_vars=[]
table_vars=[]
infoFileNames=[]
#create file names and var names 
for group_No in range(1,5):
    for subset_No in range(1,51):
        fileName='group'+str(group_No)+'_subset_'+str(subset_No)+'.mat'
        tableFileName='group'+str(group_No)+'_waveformTableSubset_'+str(subset_No)+'.csv'
        fileNames.append(Path(RFDatasetDir+'/'+'Group'+str(group_No)+'/'+fileName))
        waveform_vars.append('group'+str(group_No)+'_waveformSubset_'+str(subset_No))
        status_vars.append('group'+str(group_No)+'_radarStatusSubset_'+str(subset_No))
        #table_vars.append('group'+str(group_No)+'_waveformTableSubset_'+str(subset_No))
        infoFileNames.append(Path(RFDatasetDir+'/'+'Group'+str(group_No)+'/'+'group'+str(group_No)+'_subset_CSVInfo'+'/'+tableFileName))
#%%
# split data into 3 groups train, validate, test
N=len(fileNames)
trainNo=int(np.round(N*0.35))
valNo=int(np.round(N*0.15))
testNo=int(np.round(N*0.50))

train_fileNames, val_fileNames, test_fileNames =np.split(fileNames,[trainNo,testNo])
train_waveform_vars, val_waveform_vars, test_waveform_vars =np.split(waveform_vars,[trainNo,testNo])
train_status_vars, val_status_vars, test_status_vars =np.split(status_vars,[trainNo,testNo])
#train_table_vars, val_table_vars, test_table_vars =np.split(fileNames,[table_vars,testNo])
train_infoFileNames, val_infoFileNames, test_infoFileNames =np.split(infoFileNames,[trainNo,testNo])

#%%
def create_full_spectro_dataset(outputFile,infoFileNames,fileNames,waveform_vars,status_vars):
    # Generate spectrogram datasets
    columnNames=pd.read_csv(infoFileNames[0]).keys()
    setInfo=pd. DataFrame(columns=columnNames)

    Nfft=128        
    NPerSeg=128    
    groupby=60      # no. of consectuive FFTs over which to take max
    NOverlap=24     

    sigLen=800000
    spectrotDim=int(((sigLen-NOverlap)//(NPerSeg-NOverlap))//groupby)
    h5Dataset=h5py.File(outputFile,'a')
    h5Dataset.create_dataset('spectroData', (0,Nfft,spectrotDim,1),dtype='uint16', maxshape=(None,Nfft,spectrotDim,1),chunks=(200,Nfft,spectrotDim,1))
    h5Dataset.create_dataset('spectroLabel', (0,1),dtype='uint16', maxshape=(None,1),chunks=(200,1))

    for J in range(len(fileNames)):
        spectroData=[]
        spectroLabel=[]
        h5pyObj = h5py.File(fileNames[J],'r')
        subsetSignals=h5pyObj[waveform_vars[J]][()].view(np.complex)
        subsetRadarStatus = h5pyObj[status_vars[J]][()]
        subsetInfo = pd.read_csv(infoFileNames[J])
        setInfo=setInfo.append(subsetInfo)
        for I in range(subsetSignals.shape[0]):
            sigIndex=int(I)
            f, t0, S0 = signal.spectrogram(subsetSignals[sigIndex], fs=10e6, nperseg=NPerSeg,nfft=Nfft,noverlap=NOverlap, scaling='spectrum', return_onesided=False)

            L = int(S0.shape[1]//groupby)
            S1 = np.reshape(S0[:,:L*groupby], (Nfft,L,groupby))
            S = np.amax(S1, axis=-1)
            # t = t0[groupby-1::groupby]
            minS=S.min()
            maxS=S.max()
            Sm=(S-minS)/(maxS-minS)

            spectroData.append(Sm)
            spectroLabel.append(subsetRadarStatus[I])
        
        spectroLabel=np.asarray(spectroLabel)
        spectroData=np.asarray(spectroData)
        spectroData=(pow(2,16)*spectroData).astype('uint16')

        h5Dataset["spectroData"].resize((h5Dataset["spectroData"].shape[0] + spectroData.shape[0]), axis = 0)
        h5Dataset["spectroData"][-spectroData.shape[0]:] = np.expand_dims(spectroData,axis=3)

        h5Dataset["spectroLabel"].resize((h5Dataset["spectroLabel"].shape[0] + spectroLabel.shape[0]), axis = 0)
        h5Dataset["spectroLabel"][-spectroLabel.shape[0]:] = spectroLabel
    
    h5Dataset.close()
    #save set info in the same dataset file
    #colMap=['BinNo', 'ChirpDirection', 'SamplingFrequency', 'PhaseCodingType','SUID', 'radarStatus', 'NoisePowerdBmPerMHz']
    # colMap=['BinNo', 'ChirpDirection', 'PhaseCodingType','SUID']
    # setInfo.loc[:,colMap] = setInfo[colMap].applymap(str)
    setInfo.to_hdf(outputFile, key='setInfo', mode='a')
    h5Dataset.close()
    #save set info in a separate csv file
    csvFile=Path(str(outputFile.parent)+'/'+outputFile.stem+'.csv')
    setInfo.to_csv(csvFile, index = False,na_rep='nan', header=True)

#%%
outFolder='../Dataset/SpectrogramMaxHoldData'

train_outputFile=Path(outFolder+'/'+'train_spectroMaxHoldDataset.h5')
create_full_spectro_dataset(train_outputFile,train_infoFileNames,train_fileNames,train_waveform_vars,train_status_vars)

val_outputFile=Path(outFolder+'/'+'val_spectroMaxHoldDataset.h5')
create_full_spectro_dataset(val_outputFile,val_infoFileNames,val_fileNames,val_waveform_vars,val_status_vars)

test_outputFile=Path(outFolder+'/'+'test_spectroMaxHoldDataset.h5')
create_full_spectro_dataset(test_outputFile,test_infoFileNames,test_fileNames,test_waveform_vars,test_status_vars)
# %%
# # how to read the data
# datasetRead=h5py.File(train_outputFile,'r')
# # spectroData is very large, so you should not read it all at once
# datasetRead['spectroData'][()].shape
# datasetRead['spectroLabel'][()].shape
# setInfoRead=pd.read_hdf(train_outputFile,key='setInfo')
# %%
