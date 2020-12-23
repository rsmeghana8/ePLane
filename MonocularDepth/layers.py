from keras.engine.topology import Layer, InputSpec
import  keras.utils.conv_utils as conv_utils
import tensorflow as tf
import keras.backend as K

class BilinearUpSampling2D(Layer):
    def __init__(self, size = (2,2),data_format = None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = K.image_data_format().lower()
        self.size = size
        self.Input_spec = InputSpec(ndim = 4)

    def call(self,inputs):
        input_shape = K.shape(inputs)
        height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
        width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        return tf.image.resize(inputs,[height,width],method = tf.image.ResizeMethod.BILINEAR)
