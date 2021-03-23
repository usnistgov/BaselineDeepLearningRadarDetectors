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

import numpy as np
import tensorflow as tf
import math
import h5py

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, fileName,x_var_name,y_var_name, batch_size, #n_classes,
                 to_fit, shuffle = False):
        self.batch_size = batch_size
        
        self.x_var_name = x_var_name
        self.y_var_name = y_var_name        
        self.h5pyObjRead = h5py.File(fileName,'r')

        self.to_fit = to_fit

        self.dim = self.h5pyObjRead[self.x_var_name].shape[1:]
        self.shuffle = shuffle
        self.n = 0
        self.NumPoints=self.h5pyObjRead[self.x_var_name].shape[0]
        self.list_IDs = np.arange(self.NumPoints)
        self.on_epoch_end()
    
    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1
        
        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0
        
        return data
    
    def __len__(self):
        # Return the number of batches of the dataset
        return math.ceil(len(self.indexes)/self.batch_size)
    
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
            (index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        X = self._generate_x(list_IDs_temp)
        
        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y #, [None] # remove [None] for tensorflow 2.2
        else:
            return X
    
    def on_epoch_end(self):
        
        self.indexes = np.arange(self.NumPoints)
        
        if self.shuffle: 
            np.random.shuffle(self.indexes)
    
    def _generate_x(self, list_IDs_temp):
               
        X = np.empty((self.batch_size, *self.dim))
        
        for i, ID in enumerate(list_IDs_temp):
            
            X[i,] = self.h5pyObjRead[self.x_var_name][ID]

            # Normalize data
            #X = (X/2**16).astype('float32')
            #X = X.astype('float32')

        return (X/2**16).astype('float32') #X#X[:,:,:, np.newaxis]
    
    def _generate_y(self, list_IDs_temp):
        
        y = np.empty(self.batch_size)
        
        for i, ID in enumerate(list_IDs_temp):
            
            #y[i] = self.y_data[ID]
            y[i] =self.h5pyObjRead[self.y_var_name][ID]

        return np.expand_dims(y,axis=1)

    def reset(self):
        self.n=0

###test
# train_generator = DataGenerator(testTempFile, 'spectroData','spectroLabel', batch_size = 20,
#                                 n_classes=2, 
#                                 to_fit=True, shuffle=False)
# d, l = next(train_generator)