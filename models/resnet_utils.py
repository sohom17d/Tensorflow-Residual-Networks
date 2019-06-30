# Tensorflow Imports
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.initializers import glorot_uniform



def main_path_block(x, filters, kernel_size, padding, conv_name, bn_name, activation):
    """
    Implementation of a main path block

    Arguments:
    x ->            tensor;     Input Tensor
    filters ->      integer;    Number of filters in the convolutional layer
    padding ->      string;     Type of padding in the convolutional layer
    conv_name ->    string;     Name of the convolutional layer
    bn_name ->      string;     Name of the batch normalization layer

    Output:
    returns ->      tensor;     Downsampled tensor
    """
    
    # Convolutional Layer
    x = Conv2D(
        filters = filters,
        kernel_size = kernel_size,
        strides = (1, 1),
        padding = padding,
        name = conv_name,
        kernel_initializer = glorot_uniform(seed = 0)
    )(x)

    # Batch Normalization Layer
    x = BatchNormalization(axis = 3, name = bn_name)(x)
    
    # Activation Layer
    if activation != None:
        x = Activation(activation)(x)
    
    return  x



def identity_block(input_tensor, middle_filter_size, filters, stage, block):
    """
    Implementation of Identity Block

    Arguments:
    input_tensor ->         tensor;         Input Tensor (n, h, w, c)
    middle_filter_size ->   integer;        Size of filter in the middle convolutional block
    filters ->              integer list;   Number of filters in the convolutional blocks
    stage ->                integer;        Denotes position of the block in the network
    block ->                string;         Denotes name of the block

    Output:
    returns ->              tensor;         Output Tensor (n, h, w, c)   
    """

    conv_name_base = 'res' + str(stage) + block + '_branch'
    batch_norm_name_base = 'batch_norm' + str(stage) + block + '_branch'

    x_shortcut = input_tensor

    # First Block of main path
    x = main_path_block(
        input_tensor,
        filters[0],
        (1, 1),
        'valid',
        conv_name_base + '2a',
        batch_norm_name_base + '2a',
        'relu'
    )

    # Middle Block of main path
    x = main_path_block(
        x,
        filters[1],
        (
            middle_filter_size,
            middle_filter_size
        ),
        'same',
        conv_name_base + '2b',
        batch_norm_name_base + '2b',
        'relu'
    )

    # Third Block of main path
    x = main_path_block(
        x,
        filters[2],
        (1, 1),
        'valid',
        conv_name_base + '2c',
        batch_norm_name_base + '2c',
        'relu'
    )

    # Skip Connection
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x



