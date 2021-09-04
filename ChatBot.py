import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)


from tensorflow.python.keras.engine.training import Model
from tensorflow.keras.layers import Input,LSTM,Dense
import numpy as np
from model import LSTMmodel
from Vocabulary import Vocabulary
import re
import os


class ChatBot():
    def __init__(self) -> None:
        """Initialize input vocabularies from previous made models
        """        
        self.input_vocab= pd.read_csv('data/input_vocabulary.csv').to_dict()['word']
        self.reverse_input_vocab={value:key for key,value in self.input_vocab.items()} 
        
        self.target_vocab=pd.read_csv('data/target_vocabulary.csv').to_dict()['word']
        self.reverse_target_vocab={value:key for key,value in self.target_vocab.items()} 
       
        self.model=tf.keras.models.load_model('models/LSTM_model_1.h5')
        self.step_by_step_model() 

        
        
    def step_by_step_model(self):
        """ Model has to deal with words that are not in input dictionary
            That's why Model can't decodes sentences using teacher forcing.
            it decodes step by step 
        """        
        # SET UP ENCODER MODEL
        self.encoder_input = self.model.input[0]
        self.encoder_output,state_h,state_c= self.model.layers[2].output
        self.encoder_states = [state_h,state_c]
        self.encoder_model = Model(self.encoder_input,self.encoder_states)
        #SET UP DECODER MODEL 
        latent_dim = 256
        decoder_state_input_hidden = Input(shape=(latent_dim,))
        decoder_state_input_cell = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
        
        # Initialize decoder layers
        decoder_inputs = Input(shape=(None,1946))
        decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
        decoder_dense = Dense(20, activation='softmax')
    
        decoder_outputs, state_hidden, state_cell = decoder_lstm( decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_hidden, state_cell]
        decoder_outputs = decoder_dense(decoder_outputs)  
          
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        

    def sentence_to_matrix(self,sentence)->np.array:
        """Make Matrix from input sentence.

        Args:
            sentence ([str]): input sentence

        Returns:
            [np.array]: [description]
        """        
        sentence =  Vocabulary.normalize(sentence)  
        tokens = sentence.split(' ')
        user_input_matrix = np.zeros((1, 12, 1991),dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in self.reverse_input_vocab:
                user_input_matrix[0, timestep, self.reverse_input_vocab[token]] = 1.
        return user_input_matrix
    
    
    def generate_response(self, user_input)->str:
        """Recieve input sentence in str, translate it to numerical data as numpy matrix and
            decode model sentence.

        Args:
            user_input ([str]): user input

        Returns:
            [str]: model output sentence
        """        
        input_matrix = self.sentence_to_matrix(user_input)
        chatbot_response = self.get_decoded_sentence(input_matrix)
        chatbot_response = chatbot_response.replace("<BOS>",'').replace("<EOS>",'')
        
        return chatbot_response
    
    def get_decoded_sentence(self,expression):
        """
        -Encode the input sentence and retrieve the initial decoder state
        -Run  steps of the decoder, The output will be the next target word.
        -Append the target word predicted and repeat till EOS, or exceedig the max sentence length 

        Args:
            expression ([np.array]): Array representing input sentence

        Returns:
            [str]: Chatbot repsnonse
        """        
        states_value = self.encoder_model.predict(expression)        
        # Initialize matrix for target sequence
        target_seq = np.zeros((1, 1, 1946))
        target_seq[0, 0, self.reverse_target_vocab['<BOS>']] = 1.
        decoded_sentence = ''
        stop_condition = False
        
        
        while not stop_condition:
            # Predict target sentence
            output_tokens, state_h, state_c = self.decoder_model.predict([target_seq] + states_value)
            # Pick most probable token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            # using save dict set token to word
            sampled_token = self.target_vocab[sampled_token_index]
            decoded_sentence += " " + sampled_token
            
            # If there is end of sentence token, or reponse is long enough end sentence
            if (sampled_token == '<EOS>' or len(decoded_sentence) > 20):
                stop_condition = True
           
            target_seq = np.zeros((1, 1, 1946))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [state_h, state_c]
            
        return decoded_sentence
    
    
    
    
if __name__ == '__main__':
    bot = ChatBot()
    x=bot.generate_response('Can we make this quick?')
    print(x)