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
# Author: Raied Caromi << raied.caromi@nist.gov >>
#%%

import h5py
from pathlib import Path

#import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.io as sio
import numpy as np
import pandas as pd

######### config local lib path
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
######### end path config
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
from lib.models import Models

#%%
# set which model to train and test: True or False 
modelsToTrain={
    "CNN3-SpectroMaxHold": False,
    "ResNet50-SpectroMaxHold": True,
    "Xception-SpectroMaxHold": False,
    "MobileNetV2-SpectroMaxHold": False
}

#%%
def tain_model(modelType, inputShape, train_spectroData, train_spectroLabel, val_spectroData, val_spectroLabel, epochs, batchSize):

    model=Models().createModel(modelType,inputShape)
    print(model.summary())
    print('Training results for model: ', modelType)

    model.fit(x=train_spectroData, y=train_spectroLabel, validation_data=(val_spectroData, val_spectroLabel), epochs=epochs, verbose=2, batch_size=batchSize)

    print('=========================================================')
    
    return model

#%%
def test_prediction(model, test_spectroData, modelType, test_spectroLabel, resultsPath, testSetInfo):
    print('Testing results for model: ', modelType)
    YOutput_p=model.predict(test_spectroData)
    YOutputLogical=YOutput_p>0.5
    YOutputInt=YOutputLogical.astype(int)
    acc=accuracy_score(test_spectroLabel,YOutputInt)   
    cm = confusion_matrix(test_spectroLabel,YOutputInt)
    print('Test accuracy=',acc)
    print('Confusion matrix')
    print(cm)
    print('=========================================================')
    matDict = {}

    matDict['test_spectroLabel'] =test_spectroLabel

    matDict['testSetInfo'] ={name: col.values for name, col in testSetInfo.items()}
    #outVar=modelType+'_out_p'
    outVar=modelType.replace('-','_')+'_out_p'

    matDict[outVar] =YOutput_p

    #output data information 
    
    resultFileName=modelType
    suffix='V1'
    sio.savemat(Path(resultsPath)/(resultFileName+'_dataOut_'+suffix+'.mat'), matDict)

    model.save(Path(resultsPath)/(modelType+'_model_'+suffix+'.h5'))
#%%
datasetFolder='../Dataset/SpectrogramMaxHoldData'

train_datasetFile=Path(datasetFolder+'/'+'train_spectroMaxHoldDataset.h5')
val_datasetFile=Path(datasetFolder+'/'+'val_spectroMaxHoldDataset.h5')
test_datasetFile=Path(datasetFolder+'/'+'test_spectroMaxHoldDataset.h5')

resultsPath='../results'
#%%
batchSize=20
epochs=50
input_shape=(128, 128, 1)
h5pyObjRead = h5py.File(train_datasetFile,'r')
train_spectroData=(np.expand_dims(h5pyObjRead['spectroData'][()],axis=3)/2**16).astype('float32')
train_spectroLabel=h5pyObjRead['spectroLabel'][()]
h5pyObjRead.close()

randIndex=np.arange(train_spectroLabel.shape[0])
np.random.shuffle(randIndex) # shuffles but return none
train_spectroData=train_spectroData[randIndex]
train_spectroLabel=train_spectroLabel[randIndex]

h5pyObjRead = h5py.File(val_datasetFile,'r')
val_spectroData=(np.expand_dims(h5pyObjRead['spectroData'][()],axis=3)/2**16).astype('float32')
val_spectroLabel=h5pyObjRead['spectroLabel'][()]
h5pyObjRead.close()

randIndex=np.arange(val_spectroLabel.shape[0])
np.random.shuffle(randIndex) # shuffles but return none
val_spectroData=val_spectroData[randIndex]
val_spectroLabel=val_spectroLabel[randIndex]

h5pyObjRead = h5py.File(test_datasetFile,'r')
test_spectroData=(np.expand_dims(h5pyObjRead['spectroData'][()],axis=3)/2**16).astype('float32')
test_spectroLabel=h5pyObjRead['spectroLabel'][()]
testSetInfo=pd.read_hdf(test_datasetFile,key='setInfo')
h5pyObjRead.close()

#%%
for modelType, value in modelsToTrain.items():
    if value:
        model=tain_model(modelType, input_shape, train_spectroData, train_spectroLabel, val_spectroData, val_spectroLabel, epochs, batchSize)
        test_prediction(model, test_spectroData, modelType, test_spectroLabel, resultsPath, testSetInfo)
