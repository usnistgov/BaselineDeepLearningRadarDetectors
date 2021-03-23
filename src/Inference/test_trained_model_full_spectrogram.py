#%%
import numpy as np
import h5py
from pathlib import Path
#mport matplotlib

import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
#from confusion_matrix_pretty_print import plot_confusion_matrix_from_data
import pandas as pd
from scipy import signal
import glob
######### config local lib path
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
######### end path config
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from lib.testProcess import constructDatasetFilesVars, createFullSpectro, loadModel, generateTestResults
#%%
modelDir='../trainedModels'
modelsToTest={
    "CNN4-Spectro": True,
    "CNN5-Spectro": True,
}
foundModels=(glob.glob(str(Path(modelDir+"/*.h5"))))

# %%
modelNamePaths=[]
for modelType, value in modelsToTest.items():
    if value:
        modelNamesMatching = [s for s in foundModels if modelType in s]
        for modelNamePath in modelNamesMatching:
            modelNamePaths.append(modelNamePath)

# %%
RFDatasetDir = r'../Dataset/RFDataset/SimulatedRadarWaveforms'
resultsPath=r'../results'

groupNoStart=3
groupNoEnd=4
fileNoStart=1
fileNoEnd=50
fileNames, waveform_vars, status_vars, infoFileNames=constructDatasetFilesVars(RFDatasetDir,groupNoStart,groupNoEnd, fileNoStart,fileNoEnd)
#%%
columnNames=pd.read_csv(infoFileNames[0]).keys()

#%%
for modelNamePath in modelNamePaths:
    radarStatus=[]
    inferenceVal=[]
    setInfo=pd. DataFrame(columns=columnNames)
    model=loadModel(modelNamePath)
    for fileIndex in range(len(fileNames)):
        #spectroData=[]
        #spectroLabel=[]
        # Get waveforms from one file
        #print(J,'\n')
        h5pyObj = h5py.File(fileNames[fileIndex],'r')
        subsetSignals=h5pyObj[waveform_vars[fileIndex]][()].view(np.complex)
        subsetRadarStatus = h5pyObj[status_vars[fileIndex]][()]
        subsetInfo = pd.read_csv(infoFileNames[fileIndex])
        setInfo=setInfo.append(subsetInfo)
        for sigIndex in range(subsetSignals.shape[0]):#range(subsetSignals.shape[0]):#np.nditer(highSNRIndex):#range(subsetSignals.shape[0]):
            spectroN=createFullSpectro(subsetSignals[sigIndex])
            out=model.predict(spectroN)
            inferenceVal.append(out[0])
            radarStatus.append(subsetRadarStatus[sigIndex])
    inferenceVal=np.asarray(inferenceVal)
    radarStatus=np.asarray(radarStatus)
    generateTestResults(inferenceVal,radarStatus,setInfo,modelNamePath,resultsPath)