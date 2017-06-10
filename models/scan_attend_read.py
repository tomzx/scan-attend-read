from keras.layers import Activation
from keras.models import Sequential

from layers.attention import Attention
from layers.decoder import Decoder
from layers.encoder import Encoder
from layers.state import State

class ScanAttendRead:
    def __init__(self, input_shape):
        self.model = Sequential()
        [self.model.add(x) for x in Encoder(input_shape).model.layers]
        [self.model.add(x) for x in Attention(self.model.layers[-1].output_shape).model.layers]
        [self.model.add(x) for x in State(self.model.layers[-1].output_shape).model.layers]
        [self.model.add(x) for x in Decoder(self.model.layers[-1].output_shape).model.layers]
        self.model.add(Activation('softmax'))
