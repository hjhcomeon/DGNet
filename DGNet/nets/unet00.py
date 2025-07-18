from nets.vgg16 import VGG16
from nets.resnet50 import ResNet50
from xml.sax.handler import feature_external_ges
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv1D, Conv2D, Dense,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, BatchNormalization,
                          Reshape, multiply)
import math
import tensorflow as tf

from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K

def se_block(input_feature, ratio=16, name=""):
	channel = input_feature.shape[-1]
	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)

	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=False,
					   bias_initializer='zeros',
					   name = "se_block_one_"+str(name))(se_feature)
					   
	se_feature = Dense(channel,
					   kernel_initializer='he_normal',
					   use_bias=False,
					   bias_initializer='zeros',
					   name = "se_block_two_"+str(name))(se_feature)
	se_feature = Activation('sigmoid')(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def channel_attention(input_feature, ratio=8, name=""):
	
	channel = input_feature.shape[-1]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_one_"+str(name))
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_two_"+str(name))
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	max_pool = GlobalMaxPooling2D()(input_feature)

	avg_pool = Reshape((1,1,channel))(avg_pool)
	max_pool = Reshape((1,1,channel))(max_pool)

	avg_pool = shared_layer_one(avg_pool)
	max_pool = shared_layer_one(max_pool)

	avg_pool = shared_layer_two(avg_pool)
	max_pool = shared_layer_two(max_pool)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, name=""):
	kernel_size = 7

	cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	concat = Concatenate(axis=3)([avg_pool, max_pool])

	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False,
					name = "spatial_attention_"+str(name))(concat)	
	cbam_feature = Activation('sigmoid')(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

def cbam_block(cbam_feature, ratio=8, name=""):
	cbam_feature = channel_attention(cbam_feature, ratio, name=name)
	cbam_feature = spatial_attention(cbam_feature, name=name)
	return cbam_feature


def eca_block(input_feature, b=1, gamma=2, name=""):
	channel = K.int_shape(input_feature)[-1]
	kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
	kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
	avg_pool = GlobalAveragePooling2D()(input_feature)
	x = Reshape((-1,1))(avg_pool)
	x = Conv1D(1, kernel_size=kernel_size, padding="same", name = "eca_layer_"+str(name), use_bias=False,)(x)
	x = Activation('sigmoid')(x)
	x = Reshape((1, 1, -1))(x)
	output = multiply([input_feature,x])
	return output

def Unet(input_shape=(256,256,3), num_classes=3, backbone = "vgg"):

    inputs = Input(input_shape)

    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    feat1 = eca_block(conv1 ,name="eac1")
    up1 =  layers.Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(feat1))

    conv2 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    conv2 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    feat2 = eca_block(conv2,name="eac2")
    up2 = layers.Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(feat2))

    conv3 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(up2)
    conv3 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    feat3 = eca_block(conv3,name="eac3")
    pool1 = MaxPooling2D(pool_size=(2, 2))(feat3)
    merge7 = concatenate([feat2, pool1], axis=3)

    conv4 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv4 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    feat4 = eca_block(conv4,name="eac4")
    pool2 = MaxPooling2D(pool_size=(2, 2))(feat4)
    merge8 = concatenate([feat1, pool2], axis=3)

    conv5 = layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv5 = layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    feat5 = eca_block(conv5,name="eac5")
    merge9 = concatenate([feat1, feat5], axis=3)

    conv9 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    feat6 = eca_block(conv10,name="eac6")
    conv11 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(feat6)

    if backbone == "vgg":
        # 512, 512, 64 -> 512, 512, num_classes
        P1 = Conv2D(num_classes, 1, activation="softmax")(conv11)

    else:
        raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        
    model = Model(inputs=inputs, outputs=P1)
    return model
