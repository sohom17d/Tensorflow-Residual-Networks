from models.resnet_utils import main_path_block, convolutional_block, identity_block
from tensorflow.keras.layers import Input, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


class Resnet50:

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes
        self.model = self.build_model()

    def build_model(self):
        """
        Implementation of the Resnet50 Architecture
        """

        input_placeholder = Input(shape = self.input_shape)
        x = ZeroPadding2D((3, 3))(input_placeholder)

        # Stage 1
        x = main_path_block(x, 64, (7, 7), 'valid', 'conv1', 'bn_conv1', 'relu', (2, 2))
        x = MaxPooling2D((3, 3), strides = (2, 2))(x)

        # Stage 2
        x = convolutional_block(x, 3, [64, 64, 256], 2, 'a', 1)
        x = identity_block(x, 3, [64, 64, 256], 2, 'b')
        x = identity_block(x, 3, [64, 64, 256], 2, 'c')

        # Stage 3
        x = convolutional_block(x, 3, [128, 128, 512], 3, 'a', 2)
        x = identity_block(x, 3, [128, 128, 512], 3, 'b')
        x = identity_block(x, 3, [128, 128, 512], 3, 'c')
        x = identity_block(x, 3, [128, 128, 512], 3, 'd')

        # Stage 4
        x = convolutional_block(x, 3, [256, 256, 1024], 4, 'a', 2)
        x = identity_block(x, 3, [256, 256, 1024], 4, 'b')
        x = identity_block(x, 3, [256, 256, 1024], 4, 'c')
        x = identity_block(x, 3, [256, 256, 1024], 4, 'd')
        x = identity_block(x, 3, [256, 256, 1024], 4, 'e')
        x = identity_block(x, 3, [256, 256, 1024], 4, 'f')

        # Stage 5
        x = convolutional_block(x, 3, [512, 512, 2048], 5, 'a', 2)
        x = identity_block(x, 3, [512, 512, 2048], 5, 'b')
        x = identity_block(x, 3, [512, 512, 2048], 5, 'c')
        
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

        return Model(input_placeholder, x, name = 'Resnet50')
