import pandas as pd
import tensorflow as tf
from fuzzywuzzy import process
from Vocabulary import Vocabulary
class ChatBot():
    def __init__(self) -> None:
        self.input_vocab= pd.read_csv('data/input_vocabulary.csv').to_dict()['word']
        self.target_vocab=pd.read_csv('data/target_vocabulary.csv').to_dict()['word']
        print(self.input_vocab)
        self.model=tf.keras.models.load_model('LSTM_model_1.h5')
    def get_expression(self,expression):
        self.expression=Vocabulary.normalize(expression)
        print(self.expression)
        if expression not in self.input_vocab.values():
            fuzzy_exp=process.extractOne(expression,self.input_vocab.values())
            print(fuzzy_exp)
        
    def send_expression(self):
        pass
        #model
    
    
    
if __name__ == '__main__':
    bot = ChatBot()
    bot.get_expression("Whats up")