from tensorflow import keras
class models:
    
    def __init__():
        pass

    ### Define convolutional network architecture ###
    def cnn():
        # num_filters = [24,32,64,128] 
        pool_size = (2, 2) 
        kernel_size = (3, 3)  
        input_shape = (60, 41, 2)
        num_classes = 10
        keras.backend.clear_session()
        
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(24, kernel_size,
                    padding="same", input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
        model.add(keras.layers.Dropout(.2))

        model.add(keras.layers.Conv2D(32, kernel_size,
                                    padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))  
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
        model.add(keras.layers.Dropout(.2))
        
        model.add(keras.layers.Conv2D(64, kernel_size,
                                    padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))  
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
        model.add(keras.layers.Dropout(.2))
        
        model.add(keras.layers.Conv2D(128, kernel_size,
                                    padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))  
        model.add(keras.layers.Dropout(.2))

        model.add(keras.layers.GlobalMaxPooling2D())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dropout(.2))

        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dropout(.2))

        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        model.compile(optimizer=keras.optimizers.Adam(1e-4), 
            loss=keras.losses.SparseCategoricalCrossentropy(), 
            metrics=["accuracy"])
            
        return model

    def crnn():
        # num_filters = [24,32,64,128] 
        pool_size = (2, 2) 
        kernel_size = (3, 3)  
        input_shape = (60, 41, 2)
        num_classes = 10
        keras.backend.clear_session()
        
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(24, kernel_size,
                    padding="same", input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
        model.add(keras.layers.Dropout(.2))

        model.add(keras.layers.Conv2D(32, kernel_size,
                                    padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))  
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
        model.add(keras.layers.Dropout(.2))
        
        model.add(keras.layers.Conv2D(64, kernel_size,
                                    padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))  
        model.add(keras.layers.MaxPooling2D(pool_size=pool_size))
        model.add(keras.layers.Dropout(.2))
        
        model.add(keras.layers.Conv2D(128, kernel_size,
                                    padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))  
        model.add(keras.layers.Dropout(.2))

        # # model.add(keras.layers.GlobalMaxPooling2D())
        # (None, 7, 5, 128)
        # (batch_size, timesteps, input_dim)
        model.add(keras.layers.Reshape((35,128), input_shape=(7,5,128)))
        input_shape = (35, 128)
        model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(keras.layers.Dropout(.2))
        model.add(keras.layers.LSTM(128))
        model.add(keras.layers.Dropout(.2))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        model.compile(optimizer=keras.optimizers.Adam(1e-4), 
            loss=keras.losses.SparseCategoricalCrossentropy(), 
            metrics=["accuracy"])
            
        return model
