import pandas as pd
from datetime import datetime



Movie_lines={}
Movie_conversations={}
input=[]
target=[]

def Movie_lines_load():
	MOVIE_LINES_COLUMNS = ['LineID', 'CharID', 'MovieID', 'CharName', 'text']
	with open('data/Cornell/cornell movie-dialogs corpus/movie_lines.txt', 'r') as file:
		data=file.read()
		data=data.split('\n')
		for line in data:
			try:
				line = line.split('+++$+++')
				temp={}
				for index,column in enumerate(MOVIE_LINES_COLUMNS):
					temp[column]=line[index].strip()
				Movie_lines[temp['LineID']]=temp

			except:
				pass
def Conversations_load():
	MOVIE_COSERVATIONS_COLUMNS=['Char1ID','Char2ID','MovieID','LinesIDs']
	with open('data/Cornell/cornell movie-dialogs corpus/movie_conversations.txt', 'r') as file:
		data = file.read()
		data = data.split('\n')
		for number,line in enumerate(data):
			try:
				line = line.split('+++$+++')
				temp = {}
				for index, column in enumerate(MOVIE_COSERVATIONS_COLUMNS):
					temp[column] = line[index].strip()
				temp['LinesIDs']=temp['LinesIDs'].replace("'",'').strip(' ][')
				temp['LinesIDs']=temp['LinesIDs'].split(',') # Last element of dict has to be list, not str
				Movie_conversations['ConvID' + str(number)]=temp
			except:
				pass
def Extracting_Lines():
	Temp_lines=Movie_lines
	Temp_conv=Movie_conversations
	start_time = datetime.now()
	print('Extracting lines...')
	deleted=0
	for ID_global,dialog in Temp_conv.items():
		for index,IDs in enumerate(dialog['LinesIDs']):
			line=find_line(IDs.strip(), Temp_lines)
			if index == len(dialog['LinesIDs'])-1 and not (index+1) %2==0: # Skip last element if its odd
				deleted+=1
				break
			if (index+1) % 2 == 0 :
				target.append({'predictID': IDs ,'predict_t': line })
				continue
			input.append({'trainID': IDs,'train_t': line })
	end_time = datetime.now()
	print('Operation Complete, Duration: {}'.format(end_time - start_time))
	print('Training data length: '+str(len(input)) )
	print('Predictions data length: '+str(len(target)) )
	print('Deleted sentences: '+str(deleted))
def find_line(IDs, Temp_lines):
	for ID,line in Temp_lines.items():
		if IDs == line['LineID']:
			line=line['text']
			del Temp_lines[ID]
			return line
def Load_data():
	Movie_lines_load()
	Conversations_load()
def data_csv():
	df=pd.DataFrame(input)
	df1=pd.DataFrame(target)
	print(df.head())
	pd.concat([df, df1], axis=1).to_csv('cornell movie-dialogs corpus/Data_formatted.csv',sep='\t')
def open_data():
	df=pd.read_csv('data/Cornell/cornell movie-dialogs corpus/Data_formatted.csv', sep='\t')
	Target=df['predict_t']
	Input = df['train_t']
	return Input, Target



