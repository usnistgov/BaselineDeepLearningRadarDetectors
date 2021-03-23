import numpy as np
from pathlib import Path

import tensorflow as tf
from scipy import signal

def constructDatasetFilesVars(RFDatasetDir,groupNoStart,groupNoEnd, fileNoStart,fileNoEnd):
    fileNames=[]
    waveform_vars=[]
    status_vars=[]
    #table_var=[]
    infoFileNames=[]
    #create file names and var names

    for group_No in range(groupNoStart,groupNoEnd+1):
        for subset_No in range(fileNoStart,fileNoEnd+1):
            groupFil='group'+str(group_No)
            GroupFol='Group'+str(group_No)
            subset=str(subset_No)
            fileName=groupFil+'_subset_'+subset+'.mat'
            tableFileName=groupFil+'_waveformTableSubset_'+subset+'.csv'
            fileNames.append(Path(RFDatasetDir+'/'+GroupFol+'/'+fileName))
            waveform_vars.append(groupFil+'_waveformSubset_'+subset)
            status_vars.append(groupFil+'_radarStatusSubset_'+subset)
            #table_var.append(group+'_waveformTableSubset_'+subset)
            infoFileNames.append(Path(RFDatasetDir+'/'+GroupFol+'/'+groupFil+'_subset_CSVInfo'+'/'+tableFileName))
    return fileNames, waveform_vars, status_vars, infoFileNames

def createFullSpectro(inputSignal,Fs=10e6,NPerSeg=256,Nfft=256,NOverlap=24):
        
        f, t0, S0 = signal.spectrogram(inputSignal, fs=Fs, nperseg=NPerSeg,nfft=Nfft,noverlap=NOverlap, scaling='spectrum', return_onesided=False)
        minS=S0.min()
        maxS=S0.max()
        spectrogramN=(S0-minS)/(maxS-minS)
        # the full spectrogram model expects shape (None, 256, 3448, 1), 
        # spectrogram shape is (256, 3448), expand dim 0 and 3
        return np.expand_dims(spectrogramN,axis=(0,3))

def createMaxHoldSpectro(inputSignal,Fs=10e6,NPerSeg=128,Nfft=128,NOverlap=24,groupby=60):

        f, t0, S0 = signal.spectrogram(inputSignal, fs=Fs, nperseg=NPerSeg,nfft=Nfft,noverlap=NOverlap, scaling='spectrum', return_onesided=False)
        L = int(S0.shape[1]//groupby)
        S1 = np.reshape(S0[:,:L*groupby], (Nfft,L,groupby))
        S = np.amax(S1, axis=-1)
        # t = t0[groupby-1::groupby]
        minS=S.min()
        maxS=S.max()
        spectrogramMN=(S-minS)/(maxS-minS)
        
        # the maxhold spectrogram model expects shape (None,  128, 128, 1), 
        # spectrogram shape is (128, 128), expand dim 0 and 3
        return np.expand_dims(spectrogramMN,axis=(0,3))

def loadModel(fileNamePath):
    model= tf.keras.models.load_model(fileNamePath)
    return model

def generateTestResults(inferenceVal,radarStatus,setInfo,modelNamePath,resultsPath):
    predicted_out=(inferenceVal>0.5).astype('uint8')
    TP = np.sum(np.logical_and(radarStatus == 1, predicted_out == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(radarStatus == 0, predicted_out == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(radarStatus == 0, predicted_out == 1))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(radarStatus == 1, predicted_out == 0))
    TPR=TP/(TP+FN) ## PD
    TNR=TN/(TN+FP)
    FNR=FN/(FN+TP)
    FPR=FP/(FP+TN) ## FA
    accM=(TP+TN)/len(radarStatus)

    allSNR=np.unique(setInfo['SNR'][~np.isnan(setInfo['SNR'])])
    modelName=Path(modelNamePath).stem
    outTextFile=Path(resultsPath+"/"+modelName+".txt")
    with open(outTextFile, "w") as text_file:
        print("Test Results for model",modelName,file=text_file)
        print('TNR=',TNR,' FPR=',FPR,'\nFNR=',FNR, ' TPR=',TPR,file=text_file)
        print("Accuracy=",accM,file=text_file)
        for selectedSNR in allSNR:
            indexSNR=np.where(setInfo.SNR==selectedSNR)
            TP = np.sum(np.logical_and(radarStatus[indexSNR] == 1, predicted_out[indexSNR] == 1))
            FN = np.sum(np.logical_and(radarStatus[indexSNR] == 1, predicted_out[indexSNR] == 0))
            TPR=TP/(TP+FN) ## PD
            FNR=FN/(FN+TP)
            print( 'For SNR=',selectedSNR, ', FNR=',FNR, ' TPR=',TPR, file=text_file)
