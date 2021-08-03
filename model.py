from Vocabulary import Vocabulary
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense



class LSTMmodel():
    
    def __init__(self) -> None:
        vocab=Vocabulary(data_range=0.5,max_len=8,min_len=4,min_number=3) # initiate vocabulary object with specific parameteres
       	self.encoder_input,self.decoder_input,self.decoder_output =vocab.get_model_data()

        self._encoder_tokens = self.encoder_input.shape[2]
        self._decoder_tokens = self.decoder_input.shape[2]
        print(self._encoder_tokens,self._decoder_tokens)
        self._DIMENSIONALITY=256
        self._BATCH_SIZE=10
        self._EPOCHS=30
        self.encoder_decoder_layers()
           
    def encoder_decoder_layers(self):
        #Encoder
        encoder_inputs = Input(shape=(None,self._encoder_tokens))
        encoder_lstm = LSTM(  self._DIMENSIONALITY, return_state=True)
        encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
        encoder_states = [state_hidden, state_cell]
        #Decoder
        decoder_inputs = Input(shape=(None, self._decoder_tokens))
        decoder_lstm = LSTM(  self._DIMENSIONALITY, return_sequences=True, return_state=True)
        decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self._decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        training_model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        #Compiling
        training_model.compile(optimizer='adam',
                               loss='categorical_crossentropy', 
                               metrics=['accuracy'], 
                               sample_weight_mode='temporal')
        #Training
        training_model.fit([self.encoder_input, self.decoder_input],
                           self.decoder_output,
                           batch_size = self._BATCH_SIZE, 
                           epochs =  self._EPOCHS,
                           validation_split = 0.2)
        training_model.save('LSTM_model_1.h5')
        
if __name__ == '__main__':
    model = LSTMmodel()