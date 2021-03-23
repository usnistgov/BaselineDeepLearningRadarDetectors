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

import pandas as pd

######### config local lib path
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
######### end path config
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
from lib.spectroDataGenerator import DataGenerator
from lib.models import Models

#%%
# set which model to train and test: True or False 
modelsToTrain={
    "CNN4-Spectro": True,
    "CNN5-Spectro": True
}

#%%
def tain_model(modelType,inputShape,train_generator,val_generator,epochs):

    model=Models().createModel(modelType,inputShape)
    print(model.summary())
    print('Training results for model: ', modelType)

    train_generator.reset()
    val_generator.reset()

    model.fit(x=train_generator,validation_data=val_generator, epochs=epochs,verbose=2)

    print('=========================================================')
    
    return model

#%%
def test_prediction(model,test_generator,modelType,test_spectroLabel,resultsPath,testSetInfo):
    test_generator.reset()
    print('Testing results for model: ', modelType)
    YOutput_p=model.predict(test_generator)
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
    outVar=modelType+'_out_p'

    matDict[outVar] =YOutput_p

    #output data information 
    
    resultFileName=modelType
    suffix='V1'
    sio.savemat(Path(resultsPath)/(resultFileName+'_dataOut_'+suffix+'.mat'), matDict)

    model.save(Path(resultsPath)/(modelType+'_model_'+suffix+'.h5'))
#%%
batch_size=25
epochs=50
input_shape=(256,3448,1)

datasetFolder=r'../Dataset/SpectrogramData'

train_datasetFile=Path(datasetFolder+'/'+'train_spectroDataset.h5')
val_datasetFile=Path(datasetFolder+'/'+'val_spectroDataset.h5')
test_datasetFile=Path(datasetFolder+'/'+'test_spectroDataset.h5')

resultsPath='../results'

datasetRead=h5py.File(test_datasetFile,'r')
test_spectroLabel=datasetRead['spectroLabel'][()]
testSetInfo=pd.read_hdf(test_datasetFile,key='setInfo')


train_generator = DataGenerator(train_datasetFile, 'spectroData','spectroLabel', batch_size = batch_size, 
                                to_fit=True, shuffle=True)

val_generator = DataGenerator(val_datasetFile, 'spectroData','spectroLabel', batch_size = batch_size,
                                to_fit=True, shuffle=True)

test_generator = DataGenerator(test_datasetFile, 'spectroData','spectroLabel', batch_size = batch_size,
                                to_fit=False, shuffle=False)

#%%
# train, test and save models
for modelType, value in modelsToTrain.items():
    if value:
        model=tain_model(modelType,input_shape,train_generator,val_generator,epochs)
        test_prediction(model,test_generator,modelType,test_spectroLabel,resultsPath,testSetInfo)
