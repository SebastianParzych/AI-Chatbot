from Vocabulary import Vocabulary
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense



class LSTMmodel():
    
    def __init__(self,data_range=0.5,max_len=8,min_len=4,min_number=3) -> None:
        """Initialize Vocabulary object with the given parameters.
           Also set model learning hyperparameters.

        Args:
            data_range (float, optional): Scope of range, which is . Defaults to 0.5.
            max_len (int, optional): [description]. Defaults to 8.
            min_len (int, optional): [description]. Defaults to 4.
            min_number (int, optional): [description]. Defaults to 3.
        """        
        vocab=Vocabulary(data_range=data_range,
                         max_len=max_len,
                         min_len=min_len,
                         min_number=min_number) # initiate vocabulary object with specific parameteres
       	self.encoder_input,self.decoder_input,self.decoder_output =vocab.get_model_data()
        print(self.encoder_input.shape)
        print(self.decoder_input.shape)
        self._encoder_tokens = self.encoder_input.shape[2]
        self._decoder_tokens = self.decoder_input.shape[2]
        print(self._encoder_tokens,self._decoder_tokens)
        self._DIMENSIONALITY=256
        self._BATCH_SIZE=10
        self._EPOCHS=200
 
        
    def lstm_model_1(self)->None:
        """Initialize encoder - decoder model, with input shapes addapted to data
        """        
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
        
        self.model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        
        
    def model_compile(self,optimizer='adamm',loss='categorical_crossentropy',metrics=['accuracy'],sample_weight_mode='temporal')->tf.keras.Model:   
        """Compile the model with the given parameters.

        Args:
            optimizer (str, optional): optimizer. Defaults to 'adamm'.
            loss (str, optional): loss function. Defaults to 'categorical_crossentropy'.
            metrics (list, optional): Metrics. Defaults to ['accuracy'].
            sample_weight_mode (str, optional): Sample weight mode . Defaults to 'temporal'.

        Returns:
            [tf.keras.Model]: compiled model
        """        
        return self.model.compile(optimizer=optimizer,
                               loss=loss, 
                               metrics=metrics, 
                               sample_weight_mode=sample_weight_mode)
        
        
        
    def model_train(self,validation_split=0.2)->None:
        """Training and saving model

        Args:
            validation_split (float, optional): [description]. Defaults to 0.2.
        """      
        self.model.fit([self.encoder_input, self.decoder_input],
                        self.decoder_output,
                        batch_size = self._BATCH_SIZE, 
                        epochs =  self._EPOCHS,
                        validation_split = validation_split)
        
        self.model.save('models/LSTM_model_2.h5')
        
if __name__ == '__main__':
    model = LSTMmodel()
    #model.model_compile()
    #model.model_train()