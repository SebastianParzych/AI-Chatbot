from LoadLines import LoadLines
import re
class Vocabulary:
	'''

	'''
	def __init__(self,data_range=1.0,max_len=10,min_len=2,min_number=2):
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
		_load=LoadLines() # Initialize LoadLines class, to get input-target sentences
		self.Input_raw, self.Target_raw= _load.read_csv() # In order to get prepeard input-target sntences, read .csv file or run LoadLines().get_input_target()
		self.Input_set= []
		self.Target_set= []
		self.data_range= 2 * round((data_range*len(self.Target_raw))/2) # Round always to number divisible by 2
		self.word_index={'EOS':1, 'BOS':2} 
		self.index_word= {1:'EOS', 2:'BOS'}
		self.word_count= {}
		self.list_of_words=[]
		self.word_number=2 
		self.deleted=0
		self.MIN_NUMBER= min_number # if a word appears less than a MIN_NUMBER it has to be deleted, default 2
		self.MAX_LEN=max_len #the  nax lenght of a sentence, default 10
		self.MIN_LEN = min_len 
		self.vocabulary_parameters()
		self.cleaning()

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
	def initialize(self):
		self.cleaning()
		self.make_fullVocabulary(self.Target_raw)
		self.make_fullVocabulary(self.Input_raw)
		self.dut_vocabulary()
		self.deleteSentences()
		self.summaryOfCleaning()
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
				input_temp=self.normalize(input_line)
				target_temp=self.normalize(target_line)
				if self.filterLength(input_line,target_line):
					input.append(input_temp)
					target.append('BOS '+target_temp+' EOS')
					filtered_index+=1
					print(f'{filtered_index}/{len(self.Input_raw)}')
			except AttributeError:
				pass
		self.Input_raw= input
		self.Target_raw=target

	def normalize(self,s):
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
		s = re.sub(r"what's", "that is", s)
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

	def filterLength(self, input_line, target_line):
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
		print(l,'---------------',all(ele >= self.MIN_LEN and ele < self.MAX_LEN for ele in l) )
		return all(ele >= self.MIN_LEN and ele < self.MAX_LEN for ele in l) 
		
	
	def make_fullVocabulary(self, data):
		print('Making Vocabulary')
		for number,line in enumerate(data):
			try:
				for word in line.split(' '):
					if word not in self.word_index:
						self.word_count[word] = 1
						self.word_index[word]=self.word_number
						self.index_word[self.word_number] =word
						self.word_number+=1
					else:
						self.word_count[word] += 1
			except:
				pass
		print('Number of all words in vocabulary: ' + str(self.word_number))
		print('Number of word_index: ' + str(len(self.word_index)))
		print('Number of ord_index: ' + str(len(self.index_word)))
		print('-----------------------------------------------------------')

	def trim_vocabulary(self):
		for word in self.word_count.keys():
			if self.word_count[word] <= self.MIN_NUMBER:
				del self.index_word[self.word_index[word]]
				del self.word_index[word]
				self.word_number-=1
				self.deleted+=1
			else:
				self.list_of_words.append(word)

	def deleteSentences(self):
		keep_inputs= []
		keep_targets=[]
		for input_line,target_line in zip(self.Input_raw, self.Target_raw):
			keep_input= True
			keep_target=True
			for word in input_line.split(' '):
				if word not in self.list_of_words:
					keep_input= False
					break
			for word in target_line.split(' '):
				if word not in self.list_of_words:
					keep_target= False
					break
			if keep_input and keep_target:
				keep_inputs.append(input_line)
				keep_targets.append(target_line)
		self.Input_set = keep_inputs
		self.Target_set = keep_targets
	def vocab_set(self,data):
		index = 0
		index_word, word_index= {}, {}
		for sentence in data:
			for word in sentence.split():
				if word in self.word_index and word not in word_index:
					index_word[index]=word
					word_index[word]=index
					index +=1
		return index_word,word_index

	def summaryOfCleaning(self):
		print('FILTERING PARAMETRS:')
		print('Minimum number of times a word appears: '+str(self.MIN_NUMBER))
		print('Maximum number of words in one sentence: ' + str(self.MAX_LEN))
		print('--------------------------------------------------------------')
		print('RESULTS OF FILTERING DATASETS')
		print('Filtered number of training dataset rows: '+str(len(self.Input_set)))
		print('Filtered number of prediction dataset rows:  ' + str(len(self.Target_set)))
		print('Number of all words in vocabulary:  ' + str(self.word_number))
		print('Number of deleted words from default  vocabulary:  '+ str(self.deleted))
		print('Deleted sentences due  to rare words:   '+str(len(self.Input_raw)-len(self.Target_set)))
		print('--------------------------------------------------------------')



if __name__ == '__main__':
    vocab=Vocabulary(data_range=1,max_len=10,min_len=2,min_number=2)
