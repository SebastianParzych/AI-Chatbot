import LoadLines
import re
class Vocabulary:
	def __init__(self):
		self.Input_raw, self.Target_raw= LoadLines.open_data()
		self.Input_set= []
		self.Target_set= []
		self.data_range=0
		self.word_index={}
		self.index_word= {-1:'EOS', -2:'BOS'}
		self.word_count= {}
		self.list_of_words=[]
		self.word_number=2
		self.deleted=0
		self.MIN_NUMBER=2 # if word appears less than MIN_NUMBER it has to be deleted, default 2
		self.MAX_LEN=10 # Max lenght of sentence, default 10
	def Initialize(self, max_len,min_number,data_range):
		self.MAX_LEN=max_len
		self.MIN_NUMBER=min_number
		self.data_range= 2 *  round((data_range*len(self.Target_raw))/2) # Round always to number divisible by 2
		self.Cleaning()
		self.make_fullVocabulary(self.Target_raw)
		self.make_fullVocabulary(self.Input_raw)
		self.Cut_vocabulary()
		self.DeleteSentences()
		self.SummaryOfCleaning()
	def Cleaning(self):
		input,target = [],[]
		for number,(input_line,target_line) in enumerate(zip(self.Input_raw, self.Target_raw)):
			if number >= self.data_range:
				break
			try:
				input_temp=self.normalize(input_line)
				target_temp=self.normalize(target_line)
				if self.filterLength(input_line,target_line):
					input.append(input_temp)
					target.append('BOS '+target_temp+' EOS')
					number+=1
					print(str(self.index)+'/'+str(len(self.Input_raw)))
			except:
				pass
		self.Input_raw= input
		self.Target_raw=target
	def normalize(self,s):
		s = s.lower().strip()
		s=re.sub(r'[\W_]+', ' ', s)
		s = re.sub(r'[0-9]+', ' ', s)
		s=s.strip()
		return s
	def filterLength(self, input_line, target_line):
		if (len(input_line.split(' ')) and len(target_line.split(' ')))<self.MAX_LEN:
			return True
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
	def Cut_vocabulary(self):
		for word in self.word_count.keys():
			if self.word_count[word] <= self.MIN_NUMBER:
				del self.index_word[self.word_index[word]]
				del self.word_index[word]
				self.word_number-=1
				self.deleted+=1
			else:
				self.list_of_words.append(word)
	def DeleteSentences(self):
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
	def Vocab_set(self,data):
		index = 0
		index_word, word_index= {}, {}
		for sentence in data:
			for word in sentence.split():
				if word in self.word_index and word not in word_index:
					index_word[index]=word
					word_index[word]=index
					index +=1
		return index_word,word_index

	def SummaryOfCleaning(self):
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


