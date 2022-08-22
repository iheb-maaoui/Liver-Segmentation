from tqdm import tqdm
import numpy as np
import cv2
import h5py
import os
import matplotlib.pyplot as plt
from glob import glob
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, EarlyStopping
from model import unet_2classes
from losses import dice_loss
import keras.backend as K
import pandas as pd
from tensorflow import keras


def SelectPatientsTrainVal(input_path, val_split):
	hf = h5py.File(input_path, "r")
	PatientsId = hf['patient_id'][0]
	print("patientId shape ", PatientsId.shape)
	
	np.random.shuffle(PatientsId)
	NPatients = PatientsId.shape[0]
	PatientsIdTrain = PatientsId[:int((1 - val_split) * NPatients + 0.5)]
	PatientsIdVal = PatientsId[int((1 - val_split) * NPatients + 0.5):]
	
	hf.close()
	return np.array(PatientsIdTrain), np.array(PatientsIdVal)

def ExtractXY(input_path, PatientsRead, image_size, num_classes, interpolation=cv2.INTER_NEAREST, ShowProgress=False,
		shuffle=False, use_3D=False, mode_3D=None, n_slice_3D=None, stride_z_3D=None):
	hf = h5py.File(input_path, "r")
	X = []
	Y = []
	Temp0 = hf['pixel_size_original'][:]
	if ShowProgress:
		LoopArray = tqdm(range(len(PatientsRead)))
	else:
		LoopArray = range(len(PatientsRead))
	for IndexPatientId in LoopArray:
		PatientId = PatientsRead[IndexPatientId]
		try:
			Temp = hf[PatientId + '_dic_msk'][:]
			if use_3D:
				pass
		except:
			print('COULD NOT READ PATIENT ', PatientId)
			continue
		
		try:
			for SliceDicom, SliceMask in zip(Temp[0], Temp[1]):
				X.append(cv2.resize(SliceDicom, (image_size, image_size), interpolation=cv2.INTER_NEAREST))
				Y.append(cv2.resize(SliceMask, (image_size, image_size), interpolation=cv2.INTER_NEAREST))
		except:
			pass
	Temp = None
	Temp1 = None
	hf.close()

	Y = np.array(Y)
	X = np.array(X)
	print("X", type(X), X.shape)
	print("Y", type(Y), Y.shape)
	
	X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1).astype('float32')
	Y = Y.reshape(Y.shape[0],Y.shape[1],Y.shape[2],1).astype('float32')
	Y=Y/255.0 
	X = (X+1000)/2000.0
	
	IndexShuffle = np.arange(Y.shape[0])
	if shuffle:
		np.random.shuffle(IndexShuffle)
		X = X[IndexShuffle]
		Y = Y[IndexShuffle]
	
	
	print("X expanded", type(X), X.shape)
	print("Y expanded", type(Y), Y.shape)
	return X, Y



def main():

	input_path = '/home/liver/Liver_Dataset_Small_Version.hdf5'
	PatientsIdTrain = []
	PatientsIdVal = []
	PatientsIdTrain, PatientsIdVal = SelectPatientsTrainVal(input_path,0.2)
	
	print('Train id :',PatientsIdTrain)
	print('Val Id :', PatientsIdVal)
	
	X_val = []
	Y_val = []
	X_val, Y_val = ExtractXY(input_path, PatientsRead=PatientsIdVal, image_size=256, num_classes=3, ShowProgress=True)


	X_train = []
	Y_train = []
	X_train, Y_train = ExtractXY(input_path, PatientsRead=PatientsIdTrain, image_size=256, num_classes=3, ShowProgress=True)
	
	
	BATCH_SIZE=16
	
	
	model = unet_2classes()

	print(model.summary())
	
	NO_OF_TRAINING_IMAGES = len(X_train)
	NO_OF_VAL_IMAGES = len(X_val)
		
	checkpoint = ModelCheckpoint("binary_smallDataset_finalVersion.hdf5", monitor='loss', verbose=1, save_best_only=True)
	csv_logger = CSVLogger('binary_smallDataset_finalVersion.log', append=True, separator=';')
	earlystopping = EarlyStopping(monitor='accuracy', verbose=1, min_delta=0.005, patience=3, mode='max')

	callbacks_list = [checkpoint, csv_logger]
 
	history = model.fit(X_train, Y_train, epochs=10, batch_size=BATCH_SIZE, verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks_list)

	model.save_weights("binary_smallDataset_finalVersion.hdf5")
	
	plt.plot(history.history['loss'],color='r')
	plt.plot(history.history['val_loss'],color='b')
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('binary_smallDataset_finalVersion.png')
	plt.show()
	hist_df = pd.DataFrame(history.history)
	hist_csv_file = 'binary_smallDataset_finalVersion.csv'
	hist_df.to_csv(hist_csv_file)

  

if __name__ == '__main__':
	main()
