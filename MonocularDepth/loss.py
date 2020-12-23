import tensorflow as tf
import keras.backend as  K

def depth_loss_function(y_true,y_pred, theta = 0.1, maxDepthval = 1000.0/10.0):
    # Point wise depth
    l_depth = K.mean(K.abs(y_pred-y_true), axis = -1)

    #Edges
    dy_true ,dx_true = tf.image.image_gradients(y_true)
    dy_pred ,dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred-dy_true)+ K.abs(dx_pred-dx_true), axis = -1)

    # Stuctural similarity index
    l_ssim = K.clip((1- tf.image.ssim(y_true,y_pred,maxDepthval))*0.5,0,1)

    # Weightages to the loss values
    a1 = 1.0
    a2 = 1.0
    a3 = theta

    return (a1 * l_ssim) + (a2 * K.mean(l_edges)) + (a3 * K.mean(l_depth))