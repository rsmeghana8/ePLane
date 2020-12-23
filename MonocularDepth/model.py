import sys
import tensorflow as tf
from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate
from layers import BilinearUpSampling2D
from loss import depth_loss_function



def create_model(existing = ''):
    if len(existing) == 0:
        base_model = applications.densenet.DenseNet121(input_shape = (None,None,3),include_top = False)
        print('Base Model Loaded')

        base_model_opt_shape = base_model.layers[-1].output.shape
        decode_filters = int(base_model_opt_shape[-1])


        # Making base model trainable
        for layer in base_model.layers:
            layer.trainable = True

        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2,2),name=name+'_upsampling2d')(tensor)

            # Skip conncetion
            up_i = Concatenate(name = name+'_concat')([up_i, base_model.get_layer(concat_with).output])

            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        decoder = Conv2D(filters = decode_filters, kernel_size = 1, padding = 'same',input_shape = base_model_opt_shape, name = 'conv2')(base_model.output)
        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')

        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

        model = Model(inputs=base_model.input, outputs=conv3)
    
    else:
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
    
    return model
