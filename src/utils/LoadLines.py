from os import stat
import pandas as pd
from datetime import datetime
import sys

from pandas.core.frame import DataFrame

class LoadLines():
	'''
	Extctract important data from Cornell Movie-Dialogs Corups
	Split lines to input and target containers.
	'''
	def __init__(self)->None:
		self.movie_lines = []
		self.movie_conversations=[]
		self.input=[]
		self.target=[]
  
  
	def get_input_target(self)->tuple([list,list]):
		'''
		Get input and target senteces list. ( In case of not having .csv file)
		'''
		return self.input , self.target


	def write_csv(self)->None:
		'''
		Write results of splitting data into input-target sets into .csv file
		'''
		df = pd.DataFrame(self.input)
		df1 = pd.DataFrame(self.target)
		pd.concat([df, df1], axis=1).to_csv('data/Data_formatted.csv', sep='\t')
  
  
	@staticmethod
	def read_csv()->pd.DataFrame:
		'''
		Read previously written .csv file and extract input-target columns
		:return: Input,Predict lines
		'''
		return pd.read_csv('data/Data_formatted.csv', sep='\t')


	def Movie_lines_load(self)->None:
		'''
		Insert into a list of dicts data from movie_lines.txt
		:return:
		'''
		MOVIE_LINES_COLUMNS = ['LineID', 'CharID', 'MovieID', 'CharName', 'text']
		with open('data/movie_lines.txt', 'r') as file:
			data=file.read()
			data=data.split('\n')
			for line in data:
				try:
					line = line.split('+++$+++')
					temp={}
					for index,column in enumerate(MOVIE_LINES_COLUMNS):
						temp[column]=line[index].strip()
					self.movie_lines.append(temp)
				except:
					pass
 

	def Conversations_load(self)->None:
		'''
		Insert into a list of dicts data from movie_conversations.txt
		:return:
		'''
		MOVIE_COSERVATIONS_COLUMNS=['Char1ID','Char2ID','MovieID','LinesIDs']
		with open('data/movie_conversations.txt', 'r') as file:
			data = file.read()
			data = data.split('\n')
			for number,line in enumerate(data):
				try:
					line = line.split('+++$+++')
					temp = {}
					for index, column in enumerate(MOVIE_COSERVATIONS_COLUMNS):
						temp[column] = line[index].strip()
					temp['LinesIDs']=temp['LinesIDs'].replace("'",'').strip(' ][')
					temp['LinesIDs']=temp['LinesIDs'].split(',') # List of Conversations ID has to be list, not string
					self.movie_conversations.append(temp)
				except:
					pass
 
 
	def Extracting_Lines(self)->None:
		'''
		Iterate through all Movie Conversations and make two sets of data: input and target.
		When a Conversation has odd number of lines, the a last line is deleted. Thanks to that
		Input and Target set has the same number of lines.
		:return:
		'''
		self.Movie_lines_load()
		self.Conversations_load()
		temp_lines=self.movie_lines.copy()
		start_time = datetime.now()
		print('Extracting lines...')
		deleted=0
		for i,dialog in enumerate(self.movie_conversations.copy()):
			LoadLines.update_progress(i/len(self.movie_conversations))
			for index, ID in enumerate(dialog['LinesIDs']):
				line=self.find_line(ID.strip(), temp_lines)
				if index == len(dialog['LinesIDs'])-1 and not (index+1) %2==0: # Skip last element if its odd
					deleted+=1
					break
				if (index+1) % 2 == 0 :
					self.target.append({'target_ID': ID ,'target_val': line })
					continue
				self.input.append({'input_ID': ID,'input_val': line })
		end_time = datetime.now()
  
  
		print('\nOperation Complete, Duration: {}'.format(end_time - start_time))
		print('Training data length: '+str(len(self.input)) )
		print('Predictions data length: '+str(len(self.target)) )
		print('Deleted sentences: '+str(deleted))
  
  
	def find_line(self,ID, temp_lines)->str:
		'''
		Find by ID specific Movie line and return its text
		:param ID: ID of movie line
		:param temp_lines: List of all lines
		:return: return text
		'''
		for line in temp_lines:
			if ID == line['LineID']:
				text=line['text']
				temp_lines.remove(line)
				return text


	@staticmethod
	def update_progress(self,progress)->None:
		'''
		Progress Bar printing
		:param progress: (current iteration)/(length of list)
		:return:
		'''
		barLength = 50
		status = ""
		if isinstance(progress, int):
			progress = round(float(progress),2)
		if progress >= 1:
			progress = 1
			status = "Done...\r\n"
		block = int(round(barLength * progress))
		text = " \rPercent: [{0}] {1}% {2}".format("█" * block + "-" * (barLength - block),
												   round(progress,2) * 100,
												   status,
												   ".2f")
		sys.stdout.write(text)
		sys.stdout.flush()
  
  
if __name__ == '__main__':
	'''
	Initial setup to write results to .csv (Estimated 50 seconds saved in future work on datasets) 
	'''
	lines = LoadLines()
	# input,target=lines.read_csv()
	# pd.set_option("display.max_rows", None, "display.max_columns", None)
	lines.Extracting_Lines()
	#lines.write_csv()