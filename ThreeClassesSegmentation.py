import h5py
import os
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.python.keras.models import load_model
from model import unet_3classes
import numpy as np
import cv2
import PIL
import PIL.Image as pil
import matplotlib.pyplot as plt
from glob import glob
from tensorflow import keras
from losses import *

def convertGray(pred_mask,dimX,dimY):
	for i in range(dimX):
		for j in range(dimY):
			if(int(pred_mask[i][j]) == 0):
				pred_mask[i][j] = int(127)
			elif( int(pred_mask[i][j]) == 1):
				pred_mask[i][j] = int(63)
			elif( int(pred_mask[i][j]) == 2):
				pred_mask[i][j] = int(0)
			else:
				print('haha')

	return pred_mask	

	
def find_between( s, first, last):
	try:
		start = s.rindex( first ) + len( first )
		end = s.rindex( last, start )
		return s[start:end]
	except ValueError:
		return ""

def convertStringToNumberIfPosible(value):

	theValue = value

	if len(value) >1 and value.startswith('0') and not value.startswith('0.'):
		return theValue

	try:
		theValue = int(value)
		return theValue
	except:
		try:
			theValue = float(value)
			return theValue
		except:
			return theValue

		
def read_ahu(fhufile, begin, end):

	with open(fhufile, encoding='ISO-8859-1') as fhuf:

		assert fhuf.readline() == 'HU\n'

		infoFHU = {}

		while (True):
			line = fhuf.readline()
			if line.startswith("#"):
				label = find_between(line, "#", ": ")

				tmpStr = "#" + label + ": "
				val = line.replace(tmpStr, "")
				val = val.replace("\n", "")
				val = convertStringToNumberIfPosible(val)
				 
				infoFHU[label] = val
			else:
				(width, height) = [int(i) for i in line.split()]
				break

		depth = int(fhuf.readline())
		infoFHU['depth'] = depth

		raster = []
		for y in range(height):
			row = []
			line = fhuf.readline()
			content_line = line.split(";")
			for x in range(width):

				try:
					nr = int(content_line[x])
					if nr < -1000 or nr > 2000 :
						nr = -1000
					row.append(nr)
				except:
					print("ERROR READING {} - X = {} ; Y = {} ------- Skip slice!".format(fhufile, x, y))
					return None

			raster.append(row)

	data = np.clip(raster, begin, end)
	infoFHU['data'] = data

	return width , height,  infoFHU

def main():
	
	
	model = unet_3classes()
	model.load_weights("B3classes_directTraining4.hdf5")
	print(model.get_weights())

	
	type = '_BCE_nBN'
	
	
	beginHisto = -1000
	endHisto = 1000
	
	folder_name = "/home/liver/Test_Dataset"

	output_folder = 'Output_Dataset_3classes_ThirdTry'
	
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
		

	
	for filename in os.listdir(folder_name):
	#for filename in ListFiles:
		if filename[-3:] == "ahu":
			#imgorig=cv2.imread(filename)
			dimorigx, dimorigy, infoFHU = read_ahu(folder_name+'/'+filename,beginHisto, endHisto)
			name = filename.replace('.ahu','')
			print('Processing', filename)
			
			imgorig = np.array(infoFHU['data']).astype(np.int16)
			#print('shape ini imgorig=',imgorig.shape)
			
			img=cv2.resize(imgorig, (256,256))
			testX = img.reshape(1, 256, 256, 1).astype('float32')
			testX=(1000+testX)/2000.0
			prediction = model.predict(testX, verbose=1)
			print(prediction)
			prediction=np.argmax(prediction, axis=-1)
			prediction=prediction.reshape(256,256).astype('float32')
			
			prediction = cv2.resize(prediction, (dimorigx,dimorigy),interpolation=cv2.INTER_NEAREST)
			predMask = convertGray(prediction,dimorigx,dimorigy)

			
			prediction  = np.array(predMask)
			cv2.imwrite(output_folder+'/'+name+'_pred'+ type +'.pgm', prediction)
			print(name+'_pred'+'.pgm')
	  
if __name__ == '__main__':
	main()
