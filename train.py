### Load necessary libraries ###
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
from models import models

model_types = ['CNN', 'CRNN']
for model_type in model_types:

    ### Train and evaluate via 10-Folds cross-validation ###
    accuracies = []
    folds = np.array(['fold1','fold2','fold3','fold4',
                    'fold5','fold6','fold7','fold8',
                    'fold9','fold10'])
    load_dir = "UrbanSounds8K/processed/"
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(folds):
        x_train, y_train = [], []
        for ind in train_index:
            # read features or segments of an audio file
            train_data = np.load("{0}/{1}.npz".format(load_dir,folds[ind]), 
                        allow_pickle=True)
            # for training stack all the segments so that they are treated as an example/instance
            features = np.concatenate(train_data["features"], axis=0) 
            labels = np.concatenate(train_data["labels"], axis=0)
            x_train.append(features)
            y_train.append(labels)

        # stack x,y pairs of all training folds 
        x_train = np.concatenate(x_train, axis = 0).astype(np.float32)
        y_train = np.concatenate(y_train, axis = 0).astype(np.float32)
        
        # for testing we will make predictions on each segment and average them to 
        # produce single label for an entire sound clip.
        test_data = np.load("{0}/{1}.npz".format(load_dir,
                    folds[test_index][0]), allow_pickle=True)
        x_test = test_data["features"]
        y_test = test_data["labels"]
        
        ##===========================
        
        model_path = "./" + model_type + "/"
        log_dir= model_path + "logs/fit/" + folds[test_index][0]
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        if model_type == "CNN":
            model = models.cnn()
        if model_type == "CRNN":
            model = models.crnn()

        model.summary()
        
        model.fit(x_train, y_train, epochs = 30, batch_size = 32, verbose = 1, validation_split=0.2,
                    use_multiprocessing=True, workers=8, callbacks=[tensorboard_callback])
        
        # evaluate on test set/fold
        y_true, y_pred = [], []
        for x, y in zip(x_test, y_test):
            # average predictions over segments of a sound clip
            avg_p = np.argmax(np.mean(model.predict(x), axis = 0))
            y_pred.append(avg_p) 
            # pick single label via np.unique for a sound clip
            y_true.append(np.unique(y)[0]) 
        accuracies.append(accuracy_score(y_true, y_pred))    
        print("Fold n accuracy: {0}".format(accuracy_score(y_true, y_pred)))

        cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        cm.figure_.savefig(model_path + 'conf_mats/fold_' + str(test_index[0]) + '_acc_' + str(accuracy_score(y_true, y_pred)) + '.png',dpi=1000)    

        model.save(model_path + "saved_models/fold_" +  str(test_index[0]))


    print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))