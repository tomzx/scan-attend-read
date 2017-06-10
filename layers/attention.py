from keras.layers import LSTM, Dense
from keras.models import Sequential

class Attention:
    def __init__(self, input_shape):
        '''
        The attention layer uses the encoder feature map e_{i,j}, the attention map at time t-1 (\alpha_{t-1}) and the state at time t-1 (s_{t-1}).
        It produces z_{(i, j), t}.
        '''
        self.model = Sequential(name='attention')
        # Location (uses the previous attention map and state)
        # Content (uses the encoded features and previous state)

        # 16/32 hidden LSTM units in each direction
        self.model.add(LSTM(32*4, input_shape=input_shape))

        # dense?
        self.model.add(Dense(1))