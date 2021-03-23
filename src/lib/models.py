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

import tensorflow as tf

class Models:

    def createModel(self,modelType,inputShape):    
        model=[]
        if(modelType == 'CNN1-IQData'):
            sigLen=800000
            numChannels=1
            dense1_dim = 200
            dense2_dim = 20
            out_dim=1
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.MaxPool1D(pool_size=50, strides=50, padding='valid',input_shape=(sigLen,numChannels)))
            model.add(tf.keras.layers.Conv1D(filters=5, kernel_size=20,strides=20, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.05)))    
            model.add(tf.keras.layers.MaxPool1D(pool_size=10, strides=10, padding='valid'))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(dense1_dim, activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(tf.keras.layers.Dense(dense2_dim, activation='tanh'))
            model.add(tf.keras.layers.Dense(out_dim, activation='sigmoid'))
            
            opt=tf.keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])

        elif(modelType == 'CNN2-IQData'):
            sigLen=800000
            numChannels=1
            dense1_dim = 500
            dense2_dim = 50
            out_dim=1
            model=tf.keras.Sequential()
            model.add(tf.keras.layers.SeparableConv1D(filters=5, kernel_size=200,strides=20, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.05),input_shape=(sigLen,numChannels)))
            model.add(tf.keras.layers.Conv1D(filters=5, kernel_size=20,strides=80, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.05)))      
            model.add(tf.keras.layers.MaxPool1D(pool_size=5, strides=None, padding='valid'))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(dense1_dim, activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(tf.keras.layers.Dense(dense2_dim, activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(tf.keras.layers.Dense(out_dim, activation='sigmoid'))

            opt=tf.keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])

        elif(modelType == 'CNN3-SpectroMaxHold'):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=inputShape))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(tf.keras.layers.Conv2D(32, (3, 3)))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(tf.keras.layers.Conv2D(64, (3, 3)))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(1))
            model.add(tf.keras.layers.Activation('sigmoid'))
            model.compile(loss="binary_crossentropy",optimizer='Nadam',metrics=["accuracy"])

        elif(modelType == 'ResNet50-SpectroMaxHold'):
            model=tf.keras.applications.ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=inputShape, pooling='max', classes=1)
            idx_of_layer_to_change = -1
            model.layers[idx_of_layer_to_change].activation = tf.keras.activations.sigmoid

            # model.save('temp_model.h5')                                                                                                                                                                                       
            # model= tf.keras.models.load_model('temp_model.h5')
            # os.remove('temp_model.h5')        
            model.compile(loss="binary_crossentropy",optimizer='RMSPROP',metrics=["accuracy"])
        
        elif(modelType == 'Xception-SpectroMaxHold'):
            model=tf.keras.applications.xception.Xception(include_top=True, weights=None, input_tensor=None, input_shape=inputShape, pooling='max', classes=1)
            idx_of_layer_to_change = -1
            model.layers[idx_of_layer_to_change].activation = tf.keras.activations.sigmoid
            
            # model.save('temp_model.h5')                                                                                                                                                                                       
            # model= tf.keras.models.load_model('temp_model.h5')
            # os.remove('temp_model.h5')
            model.compile(loss="binary_crossentropy",optimizer='Nadam',metrics=["accuracy"])
        
        elif(modelType == 'MobileNetV2-SpectroMaxHold'):
            model=tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights=None, input_tensor=None,input_shape=inputShape, pooling='max', classes=1)
            idx_of_layer_to_change = -1
            model.layers[idx_of_layer_to_change].activation = tf.keras.activations.sigmoid
        
            # model.save('temp_model.h5')                                                                                                                                                                                       
            # model= tf.keras.models.load_model('temp_model.h5')
            # os.remove('temp_model.h5')        
            model.compile(loss="binary_crossentropy",optimizer='Nadam',metrics=["accuracy"])
        
        elif(modelType == 'CNN4-Spectro'):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Conv2D(16, (6, 80), input_shape=inputShape))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 27)))

            model.add(tf.keras.layers.Conv2D(16, (4, 4)))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(tf.keras.layers.Conv2D(8, (4, 4)))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(1))
            model.add(tf.keras.layers.Activation('sigmoid'))

            model.compile(loss="binary_crossentropy",optimizer='RMSPROP',metrics=["accuracy"])

        elif(modelType == 'CNN5-Spectro'):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 14),input_shape=inputShape))
            model.add(tf.keras.layers.Conv2D(32, (3, 3)))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 14)))

            model.add(tf.keras.layers.Conv2D(32, (3, 3)))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(tf.keras.layers.Conv2D(64, (3, 3)))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(tf.keras.layers.Dense(64))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(1))
            model.add(tf.keras.layers.Activation('sigmoid'))

            model.compile(loss="binary_crossentropy",optimizer='RMSPROP',metrics=["accuracy"])
    
        return model
