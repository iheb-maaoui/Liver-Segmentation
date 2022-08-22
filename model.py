from tensorflow import keras
from keras.layers import Input,BatchNormalization,Conv2D,MaxPooling2D,UpSampling2D,concatenate
from keras.models import Model
from losses import dice_coef_loss,weighted_dice_loss_3classes

def unet_2classes():
	inputs = Input((256, 256, 1))
	BN0 = BatchNormalization()(inputs)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BN0)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	BN1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(BN1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	BN2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(BN2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	BN3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(BN3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	BN4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(BN4)

	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	BN5 = BatchNormalization()(conv5)
	encode = [BN1, BN2, BN3, BN4, BN5]
	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BN5)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv5))
	BN6 = BatchNormalization()(up6)
	merge6 = concatenate([encode[-2], BN6], axis=3)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv6))
	BN7 = BatchNormalization()(up7)
	merge7 = concatenate([encode[-3], BN7], axis=3)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv7))
	BN8 = BatchNormalization()(up8)
	merge8 = concatenate([encode[-4], BN8], axis=3)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv8))
	BN9 = BatchNormalization()(up9)
	merge9 = concatenate([encode[-5], BN9], axis=3)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

	model = Model(inputs, conv10)

	opt = keras.optimizers.Adam(learning_rate=1e-4)
	model.compile(loss=dice_coef_loss, optimizer=opt,metrics=['accuracy'])
	return model


def unet_3classes():
	inputs = Input((256, 256, 1))
	BN0 = BatchNormalization()(inputs)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BN0)
	conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
	BN1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(BN1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	BN2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(BN2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	BN3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(BN3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
	BN4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(BN4)

	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
	BN5 = BatchNormalization()(conv5)
	encode = [BN1, BN2, BN3, BN4, BN5]
	conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BN5)

	up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv5))
	BN6 = BatchNormalization()(up6)
	merge6 = concatenate([encode[-2], BN6], axis=3)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

	up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv6))
	BN7 = BatchNormalization()(up7)
	merge7 = concatenate([encode[-3], BN7], axis=3)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

	up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv7))
	BN8 = BatchNormalization()(up8)
	merge8 = concatenate([encode[-4], BN8], axis=3)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

	up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
		UpSampling2D(size=(2, 2))(conv8))
	BN9 = BatchNormalization()(up9)
	merge9 = concatenate([encode[-5], BN9], axis=3)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
	conv10 = Conv2D(3, (1,1), activation='softmax')(conv9)
	model = Model(inputs, conv10)

	opt = keras.optimizers.Adam(learning_rate=1e-4)
	model.compile(loss=weighted_dice_loss_3classes, optimizer=opt,metrics=['accuracy'])
	return model