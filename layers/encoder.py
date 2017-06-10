from keras.layers import Dropout, Conv2D, Dense
from keras.models import Sequential

from layers.grid_rnn import GridRNN

class Encoder:
    def __init__(self, input_shape):
        '''
        The encoder layer uses the input image pixels (features) and produces an encoded feature map e_{i, j}, where (i, j) are coordinates in the feature maps.
        '''
        self.model = Sequential(name='encoder')

        # 4, 20, 100 MDLSTM
        # 12, 32 convolution
        # Dropout after every MDLSTM
        self.model.add(GridRNN(4, input_shape=input_shape))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(12, (2, 4)))
        self.model.add(GridRNN(20))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(32, (2, 4)))
        self.model.add(GridRNN(100))
        self.model.add(Dropout(0.5))
        # 80 dense
        self.model.add(Dense(80))

    # def build(self, input_shape):
    #     pass
    #
    # def call(self, inputs, **kwargs):
    #     self.model.call(inputs, kwargs['mask'])
    #
    # def compute_output_shape(self, input_shape):
    #     return self.model.output_shape()