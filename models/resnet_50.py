from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Input, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


class Resnet50:

    
    def __init__(self, input_shape, classes):
        """
        Constructor of Resnet50

        Arguments:
        input_shape ->      tuple;      Shape of input tensor
        classes     ->      integer;    Number of classes
        """
        self.input_shape = input_shape
        self.classes = classes
        self.build_model()
    

    def main_path_block(self, x, filters, kernel_size, padding, conv_name, bn_name, activation = None, strides = (1, 1)):
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
            strides = strides,
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
    


    def identity_block(self, input_tensor, middle_filter_size, filters, stage, block):
        """
        Implementation of Identity Block

        Arguments:
        input_tensor ->         tensor;         Input Tensor (n, h, w, c)
        middle_filter_size ->   integer;        Size of filter in the middle convolutional block
        filters ->              integer list;   Number of filters in the convolutional blocks
        stage ->                integer;        Denotes position of the block in the network
        block ->                string;         Denotes name of the block

        Output:
        returns ->              tensor;         Output Tensor (h, w, c)   
        """

        # Set the naming convention
        conv_name_base = 'res' + str(stage) + block + '_branch'
        batch_norm_name_base = 'batch_norm' + str(stage) + block + '_branch'

        x_shortcut = input_tensor

        # First Block of main path
        x = self.main_path_block(
            input_tensor,
            filters[0],
            (1, 1),
            'valid',
            conv_name_base + '2a',
            batch_norm_name_base + '2a',
            'relu'
        )

        # Middle Block of main path
        x = self.main_path_block(
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

        # Last Block of main path
        x = self.main_path_block(
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
    


    def convolutional_block(self, input_tensor, middle_filter_size, filters, stage, block, stride = 2):
        """
        Implementation of Identity Block

        Arguments:
        input_tensor ->         tensor;         Input Tensor (n, h, w, c)
        middle_filter_size ->   integer;        Size of filter in the middle convolutional block
        filters ->              integer list;   Number of filters in the convolutional blocks
        stage ->                integer;        Denotes position of the block in the network
        block ->                string;         Denotes name of the block
        stride ->               integer;        Stride in the convolutional layer

        Output:
        returns ->              tensor;         Output Tensor (h, w, c)   
        """
        
        # Set the naming convention
        conv_name_base = 'res' + str(stage) + block + '_branch'
        batch_norm_name_base = 'batch_norm' + str(stage) + block + '_branch'

        x_shortcut = input_tensor

        ## MAIN PATH ##

        # First Block of Main Path
        x = self.main_path_block(
            input_tensor,
            filters[0],
            (1, 1),
            'valid',
            conv_name_base + '2a',
            batch_norm_name_base + '2a',
            'relu',
            (stride, stride)
        )

        # Middle Block of main path
        x = self.main_path_block(
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

        # Last Block of main path
        x = self.main_path_block(
            x,
            filters[2],
            (1, 1),
            'valid',
            conv_name_base + '2c',
            batch_norm_name_base + '2c',
            None
        )

        ## Skip Connection Convolutional Block ##
        x_shortcut = self.main_path_block(
            x_shortcut,
            filters[2],
            (1, 1),
            'valid',
            conv_name_base + '1',
            batch_norm_name_base + '1',
            None,
            (stride, stride)
        )

        # Skip Connection
        x = Add()([x, x_shortcut])
        x = Activation('relu')(x)

        return x



    def build_model(self):
        """
        Implementation of the Resnet50 Architecture
        """

        input_placeholder = Input(shape = self.input_shape)
        x = ZeroPadding2D((3, 3))(input_placeholder)

        # Stage 1
        x = self.main_path_block(x, 64, (7, 7), 'valid', 'conv1', 'bn_conv1', 'relu', (2, 2))
        x = MaxPooling2D((3, 3), strides = (2, 2))(x)

        # Stage 2
        x = self.convolutional_block(x, 3, [64, 64, 256], 2, 'a', 1)
        x = self.identity_block(x, 3, [64, 64, 256], 2, 'b')
        x = self.identity_block(x, 3, [64, 64, 256], 2, 'c')

        # Stage 3
        x = self.convolutional_block(x, 3, [128, 128, 512], 3, 'a', 2)
        x = self.identity_block(x, 3, [128, 128, 512], 3, 'b')
        x = self.identity_block(x, 3, [128, 128, 512], 3, 'c')
        x = self.identity_block(x, 3, [128, 128, 512], 3, 'd')

        # Stage 4
        x = self.convolutional_block(x, 3, [256, 256, 1024], 4, 'a', 2)
        x = self.identity_block(x, 3, [256, 256, 1024], 4, 'b')
        x = self.identity_block(x, 3, [256, 256, 1024], 4, 'c')
        x = self.identity_block(x, 3, [256, 256, 1024], 4, 'd')
        x = self.identity_block(x, 3, [256, 256, 1024], 4, 'e')
        x = self.identity_block(x, 3, [256, 256, 1024], 4, 'f')

        # Stage 5
        x = self.convolutional_block(x, 3, [512, 512, 2048], 5, 'a', 2)
        x = self.identity_block(x, 3, [512, 512, 2048], 5, 'b')
        x = self.identity_block(x, 3, [512, 512, 2048], 5, 'c')
        
        # Average Pooling Layer
        x = AveragePooling2D((2, 2), name = 'avg_pool')(x)
        
        # Fully Connected Layer
        x = Flatten()(x)
        x = Dense(
            self.classes,
            activation = 'softmax',
            name = 'fc_' + str(self.classes),
            kernel_initializer = glorot_uniform(seed = 0)
        )(x)

        self.model = Model(input_placeholder, x, name = 'Resnet50')
    


    def summary(self):
        self.model.summary()
