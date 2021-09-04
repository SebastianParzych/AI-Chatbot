from tensorflow.python.keras.backend import dtype
from LoadLines import LoadLines
import re
import numpy as np
import pandas as pd


class Vocabulary:
    
    
	def __init__(self,data_range=1.0,max_len=10,min_len=1,min_number=2):
		'''
		Initialize and prepeare  parameters for data processing.
		args:
			max_len: int, default = 10
				The maximum number of words in a sentence.
			min_len: int, default = 2
				The minimum number of words in a sentence.
			min_number: int, default = 2
				Word threshold value, if a word appears less than min_number, it has to be deleted.
				(Deleting words from vocabulary and also input-target sentences, in which those words appearss)
			data_range: float, default=1
				fraction of data used in learning.
		'''
		print('Vocab initialized')
		#_load=LoadLines # Initialize LoadLines class, to get input-target sentences
		self.Input_raw, self.Target_raw= LoadLines.read_csv() # In order to get prepeard input-target sntences, read .csv file or run LoadLines().get_input_target()
		self.data_range= 2 * round((data_range*len(self.Target_raw))/2) # Round always to number divisible by 2
		self.list_of_words=[]
		self.word_number=2 

		self.MIN_NUMBER= min_number # if a word appears less than a MIN_NUMBER it has to be deleted, default 2
		self.MAX_LEN=max_len #the  nax lenght of a sentence, default 10
		self.MIN_LEN = min_len 
		self.vocabulary_parameters()
		self.cleaning()
		word_count= self.count_appears(self.Target_raw+self.Input_raw)
		self.trim_sentences(word_count)
  
  
	def get_model_data(self):
		'''
		return:

  		'''
		input_index_word, input_word_index, input_max_words= self.features_set(self.Input_set,False)
		target_index_word, target_word_index, target_max_words= self.features_set(self.Target_set,True)
		self.write_to_csv_index_words(input_index_word, target_index_word)
		# Initializing empty datasets for models
		encoder_input = np.zeros((len(self.Input_set),input_max_words,len(input_index_word)),dtype='float32')
		decoder_input = np.zeros((len(self.Input_set),target_max_words,len(target_index_word)),dtype='float32')
		decoder_output = np.zeros((len(self.Target_set),target_max_words,len(target_index_word)),dtype='float32')
  
		for line,(input,target) in enumerate(zip(self.Input_set,self.Target_set)):
			for timestemp,token in enumerate(input):
				encoder_input[line,timestemp,input_word_index[token]]=1
			for timestemp,token in enumerate(target):
				decoder_output[line,timestemp,target_word_index[token]]=1
				if timestemp>0:
					decoder_input[line,timestemp-1,target_word_index[token]]=1

		return encoder_input,decoder_input,decoder_output


	def write_to_csv_index_words(self,input_index_word,target_index_word):
		''' Save vocabularies of future chatbot to csve'''
		df_input=pd.DataFrame.from_dict(input_index_word,orient='index',columns=['word'])
		df_target=pd.DataFrame.from_dict(target_index_word,orient='index',columns=['word'])
		df_input.index.name='index'
		df_target.index.name='index'
		df_input.to_csv('data/input_vocabulary.csv')
		df_target.to_csv('data/target_vocabulary.csv')
  
  
	@staticmethod
	def read_csv_index_words():
		return pd.read_csv('data/input_targete_vocabulary.csv')


	def vocabulary_parameters(self):
		'''
		Show the values of the parameters used in creating the vocabulary.
		'''
		print("-----------------------------------")
		print(f'Input and Target sentences: {len(self.Input_raw)}')
		print(f'Data range: {self.data_range} / {len(self.Input_raw)}')
		print(f'Range of number of words in a sentence: ({self.MIN_LEN},{self.MAX_LEN})')
		print(f'Mimnimal number of word appears in dataset: {self.MIN_NUMBER}')
		print("-----------------------------------")


	def cleaning(self):
		'''
		First normalization of data.
			- Normalize structure of sentences: normalize()
			- Unifying the length of the sentences. filterLength (input_line,target_line)
			- Adding to the target sentences:
				-"BOS"- Beginningo of the sentence,
				-"EOS"- End of the sentence.
		'''
		input,target = [],[]
		filtered_index = 0 # Number of filltered sentences pairs.
		for (input_line,target_line) in zip(self.Input_raw, self.Target_raw):
			if filtered_index >= self.data_range:
				break
			try:
				input_temp=Vocabulary.normalize(input_line)
				target_temp=Vocabulary.normalize(target_line)
				if self.filter_length(input_line,target_line):
					input.append(input_temp.split(' '))
					target.append(('<BOS> '+target_temp+' <EOS>').split(' '))
					filtered_index+=1
					print(f'{filtered_index}/{len(self.Input_raw)}')
			except AttributeError:
				pass
		self.Input_raw= input
		self.Target_raw=target
  
  
	@staticmethod
	def normalize(s):
		'''
		Normalization of a sentences.
		arg:
			s: str
				Input/target sentence.
		returns:
			s: str
				normalized input/target sentence.
		'''
		s = s.lower().strip()
		s = re.sub(r"i'm", "i am", s)
		s = re.sub(r"he's", "he is", s)
		s = re.sub(r"she's", "she is", s)
		s = re.sub(r"it's", "it is", s)
		s = re.sub(r"that's", "that is", s)
		s = re.sub(r"what's", "what is", s)
		s = re.sub(r"where's", "where is", s)
		s = re.sub(r"how's", "how is", s)
		s = re.sub(r"\'ll", " will", s)
		s = re.sub(r"\'ve", " have", s)
		s = re.sub(r"\'re", " are", s)
		s = re.sub(r"\'d", " would", s)
		s = re.sub(r"\'re", " are", s)
		s = re.sub(r"won't", "will not", s)
		s = re.sub(r"can't", "cannot", s)
		s = re.sub(r"n't", " not", s)
		s = re.sub(r"'bout", "about", s)
		s = re.sub(r"'til", "until", s)
		s=re.sub(r'[\W_]+', ' ', s)
		s = re.sub(r'[0-9]+', ' ', s)
		return s.strip()


	def filter_length(self, input_line, target_line):
		'''
		Unifying the length of the sentences.
		Returns True whenever  word numbers in a both sentences are intersect range (MIN_LEN,MAX_LEN).
		args:
			input_line	str
				Input sentence.
			target_line: str
				Reeponse sentence.
			return: bool
		'''
		l = [len(input_line.split(' ')), len(target_line.split(' '))]
		return all(ele >= self.MIN_LEN and ele < self.MAX_LEN for ele in l) 


	def count_appears(self, data):
		'''
		Counting the number of occurrences of a specific words in input,target sets.
		args:
			data:
				Merged input and target sentences.
  		'''
		word_count= {'<EOS>':0,'<BOS>':0}
		for line in data:
			for word in line:
				if word not in word_count:
					word_count[word] = 1
				else:
					word_count[word] += 1

		return word_count


	def trim_sentences(self,word_count):
		'''
		Delete pairs of input target sentences, when on of them contain rare words.
		args:
			word_count: dict
				Dictinary containging keys: unique words, values: number of times speciffic word occure in input,target sentences.
		'''
		keep_inputs= []
		keep_targets=[]
		self.deleted=0
		for input_line,target_line in zip(self.Input_raw, self.Target_raw):
			keep_input= True
			keep_target=True
			for word in input_line:
				if word_count[word]<=self.MIN_NUMBER:
					self.deleted+=1	
					keep_input= False
					break
			for word in target_line:
				if word_count[word]<=self.MIN_NUMBER:
					self.deleted+=1	
					keep_target= False
					break
			if keep_input and keep_target:
				keep_inputs.append(input_line)
				keep_targets.append(target_line)
    
		self.Input_set = keep_inputs
		self.Target_set = keep_targets


	def features_set(self,data,istarget):
		'''
		Creating an input features dictinaries index_word,word_index, 
  		args:
			data: list
				List of sentences
			istarget: bool
				If current dataset is target sentences.
		returns:
			index_word: dict
			word_index: dict
		'''
		max_words = 0
		index = 0
		index_word, word_index={},{}
		if istarget:
			index=2
			index_word,word_index={0:'<EOS>', 1:'<BOS>'},{'<EOS>':0, '<BOS>':1} 
     		
		for sentence in data:
			if len(sentence)>max_words:
				max_words=len(sentence)
			for word in sentence:
				if word not in word_index:
					index_word[index]=word
					word_index[word]=index
					index +=1
     
		print(max_words)
		return index_word,word_index, max_words


	def summaryOfCleaning(self):
		print('--------------------------------------------------------------')
		print('RESULTS OF FILTERING DATASETS')
		print(f'Filtered number of input sentences: {len(self.Input_set)}')
		print(f'Filtered number of target sentences: {len(self.Target_set)}')
		print(f'Number of all unique words in input sentences: {self.INPUT_VOCAB_SIZE}')
		print(f'Number of all unique words in target sentences: {self.TARGET_VOCAB_SIZE}')
		print(f'Number of deleted words from default  vocabulary: {self.deleted}')
		print(f'Deleted sentences due  to rare words: {len(self.Input_raw)-len(self.Input_set)}')
		print('--------------------------------------------------------------')


if __name__ == '__main__':
    vocab=Vocabulary(data_range=0.5,max_len=8,min_len=4,min_number=3) # initiate vocabulary object with specific parameteres
    encoder_input,decoder_input,decoder_output =vocab.get_model_data()