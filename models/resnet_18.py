from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Input, ZeroPadding2D, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


class Resnet18:

    
    def __init__(self, input_shape, classes):
        """
        Constructor of Resnet50

        Arguments:
        input_shape ->      tuple;      Shape of input tensor
        classes     ->      integer;    Number of classes
        """
        self.input_shape = input_shape
        self.classes = classes
        self.build_network()
    

    def main_path_block(self, x, filters, kernel_size, padding, conv_name, bn_name, batch_norm = True, activation = None, strides = (1, 1)):
        """
        Implementation of a main path block

        Arguments:
        x ->            tensor;     Input Tensor
        filters ->      integer;    Number of filters in the convolutional layer
        padding ->      string;     Type of padding in the convolutional layer
        conv_name ->    string;     Name of the convolutional layer
        bn_name ->      string;     Name of the batch normalization layer
        batch_norm ->   boolean;    Wheather to apply batch normalization or not
        activation ->   string;     Activation to be applied
        strides ->      tuple;      Convolutional Strides

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
        if batch_norm:
            x = BatchNormalization(axis = 3, name = bn_name)(x)
        
        # Activation Layer
        if activation != None:
            x = Activation(activation)(x)
        
        return  x



    def identity_block(self, input_tensor, filters, activation, stage, block, start_with_batch_norm = True):
        """
        Implementation of Identity Block

        Arguments:
        input_tensor ->         tensor;         Input Tensor (n, h, w, c)
        filters ->              integer;        Number of filters in the convolutional blocks
        activations ->          string list;    List of Activations
        stage ->                integer;        Denotes position of the block in the network
        block ->                string;         Denotes name of the block
        start_with_batch_norm-> boolean;        Whether to start the block with Batchnormalization or not

        Output:
        returns ->              tensor;         Output Tensor (h, w, c)   
        """

        # Set the naming convention
        conv_name_base = 'res' + str(stage) + block + '_branch'
        batch_norm_name_base = 'batch_norm' + str(stage) + block + '_branch'

        x_shortcut = input_tensor
        x = None

        if start_with_batch_norm:
            # Start with Batch Normalization
            x = BatchNormalization(axis = 3, name = batch_norm_name_base + '_1')(input_tensor)

            x = Activation(activation)(x)
        
        else:
            x = input_tensor
        
        # Main Path Block a
        x = self.main_path_block(
            x,
            filters,
            (3, 3),
            'same',
            conv_name_base + '2a',
            batch_norm_name_base + '2a',
            activation = activation
        )

        # Main Path Block b
        x = self.main_path_block(
            x, filters,
            (3, 3), 'same',
            conv_name_base + '2b',
            batch_norm_name_base + '2b',
            batch_norm = False,
            activation = None
        )

        # Add skip connection
        x = Add()([x, x_shortcut])

        return x
    

    def convolutional_block(self, input_tensor, filters, activation, stage, block, start_with_batch_norm = True):
        """
        Implementation of Convolutional Block

        Arguments:
        input_tensor ->         tensor;         Input Tensor (n, h, w, c)
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
        x = None

        ## MAIN PATH ##

        if start_with_batch_norm:
            # Start with Batch Normalization
            x = BatchNormalization(name = batch_norm_name_base + '_1')(input_tensor)

            x = Activation(activation)(x)
        
        else:
            x = input_tensor
        
        # Main path block a
        x = self.main_path_block(
            x, filters[0],
            (3, 3), 'same',
            conv_name_base + '2a',
            batch_norm_name_base + '2a',
            activation = activation,
            strides = (2, 2)
        )

        # Main path block b
        x = self.main_path_block(
            x, filters[1],
            (3, 3), 'same',
            conv_name_base + '2b',
            batch_norm_name_base + '2b',
            batch_norm = False,
            activation = None
        )

        ## SKIP PATH ##

        # Convolutional Block
        x_shortcut = self.main_path_block(
            x_shortcut, filters[2],
            (3, 3), 'same',
            conv_name_base + '2b',
            batch_norm_name_base + '2b',
            batch_norm = False,
            activation = None
        )

        # Skip Connection
        x = Add()([x, x_shortcut])

        return x
    

    def build_network(self):
        """
        Implementation of the Resnet18 Architecture
        """

        input_placeholder = Input(shape = self.input_shape)

        # Stage 1
        x = self.main_path_block(
            input_placeholder,
            64, (7, 7), 'same',
            'conv1', 'bn_conv1',
            activation = 'relu',
            strides = (2, 2)
        )
        x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)

        # Stage 2
        x = self.identity_block(x, 64, 'relu', 2, 'a', False)
        x = self.identity_block(x, 64, 'relu', 2, 'b')

        # Stage 3
        x = self.convolutional_block(x, [128, 128, 128], 'relu', 3, 'a')
        x = self.identity_block(x, 128, 'relu', 3, 'b')

        # Stage 4
        x = self.convolutional_block(x, [256, 256, 256], 'relu', 4, 'a')
        x = self.identity_block(x, 256, 'relu', 4, 'b')

        # Stage 5
        x = self.convolutional_block(x, [512, 512, 512], 'relu', 5, 'a')
        x = self.identity_block(x, 512, 'relu', 4, 'b')

        # Fully Connected Layers
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((2, 1), padding = 'valid', strides = (2, 2))(x)
        x = Flatten()(x)
        x = Dense(512)
        x = Dense(
            self.classes, activation = 'softmax',
            name = 'fc_' + str(self.classes),
            kernel_initializer = glorot_uniform(seed = 0)
        )(x)

        self.model = Model(input_placeholder, x, name = 'Resnet18')
    

    def summary(self):
        """
        Print the Model Summary
        """
        self.model.summary()