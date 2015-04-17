########################################
## MACHINE LEARNING FOR PUBLIC POLICY ##
## Assignment 1 - Student Prediction  ##
## Bridgit Donnelly					  ##
########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import requests
import json


## OUTPUT FUNCTION
def output(filename, function):
	with open(filename, 'w') as f:
		f.write(function)
		f.close()

'''
Task 1
The first task is to load the file and generate summary statistics for each field as well as 
probability distributions or histograms. The summary statistics should include mean, median, 
mode, standard deviation, as well as the number of missing values for each field.
'''

def summarize(df, num_fields, non_num_fields, filename):
	'''Given a dataframe and a list of numeric and non-numeric fields, this outputs a text file
	with relevant summary statistics.'''

	## Summary Statistics ##
	row_count = df.count().max()
	rv = ''

	# Numeric Data
	for field in num_fields:
		
		mean = df[field].mean()
		median = df[field].median()
		
		mode = [str(value) for value in df[field].mode()]
		mode = ', '.join(mode)

		std = df[field].std()
		missing_val = row_count - df[field].count()

		df[field].hist(by=df[field])

		rv += '\n{0}\nMean: {1}\nMedian: {2}\nMode: {3}\nStandard Deviation: {4}\nMissing Values: {5}\n'.format(field, mean, median, mode, std, missing_val)


	# Non-Numeric Data
	for field in non_num_fields:

		mode = [value for value in df[field].mode()]
		mode = ''.join(mode)

		missing_val = row_count - df[field].count()

		rv += '\n{0}\nMode: {1}\nMissing Values: {2}\n'.format(field, mode, missing_val)

	## Output Text File
	with open(filename, 'w') as f:
		f.write(rv)
		f.close()


def histogram(df, field):
	'''Given a dataframe a field, this creates histograms.'''

	fn = field + '.png'

	#Determine number of bins based on number of different values
	bins = len(df[field].value_counts())

	#Create histogram
	df[field].hist(xlabelsize=10, ylabelsize=10, bins=bins)
	pylab.title("Distribution of {0}".format(field))
	pylab.xlabel(field)
	pylab.ylabel("Counts")
	pylab.savefig(fn)
	pylab.close()


'''
Task 2
You will notice that a lot of students are missing gender values . Your task is to infer the 
gender of the student based on their name. Please use the API at www.genderize.io to infer the 
gender of each student and generate a new data file.
'''

## API Code adapted from https://github.com/block8437/gender.py/blob/master/gender.py
def getGenders(names):
	'''Given a list of names, returns the predicted genders'''

	url = ""
	cnt = 0
	for name in names:
		if url == "":
			url = "name[0]=" + name
		else:
			cnt += 1
			url = url + "&name[" + str(cnt) + "]=" + name
		

	req = requests.get("http://api.genderize.io?" + url)
	results = json.loads(req.text)
	return results['gender']
	
	#retrn = []
	#for result in results:
	#	print result
		#if result["gender"] is not None:
		#	retrn.append((result["gender"], result["probability"], result["count"]))
		#else:
		#	retrn.append((u'None',u'0.0',0.0))
	#return retrn


def genderize(df):
	'''This takes a dataframe, replaces missing gender information with predicted values from
	genderize.io and returns an updated dataframe'''

	for index, row in df.iterrows():
		if pd.isnull(row['Gender']):
			name = [row['First_name']]
			gender = getGenders(name).title()
			df.ix[index, 'Gender'] = gender

	return df
		

'''
Task 3
You will also notice that some of the other attributes are missing. Your task is to fill in the 
missing values for Age, GPA, and Days_missed using the following approaches: 
* Fill in missing values with the mean of the values for that attribute
* Fill in missing values with a class-conditional mean (where the class is whether they graduated 
	or not).
* Is there a better, more appropriate method for filling in the missing values? If yes, describe 
	and implement it. 
You should create 2 new files with the missing values filled, one for each approach A, B, and C 
and submit those along with your code.
'''

def replace_with_mean(df, fields):
	'''Given a dataframe and fields, replaces null values for each field with its respective
	mean and returns an updated dataframe.'''

	# Find means for each field
	for field in fields:
		mean = df[field].mean()

		# Replace Null Values
		for index, row in df.iterrows():
			if pd.isnull(row[field]):
				df.ix[index, field] = mean

	return df


def replace_by_graduated(df, fields):
	'''Given a dataframe and fields, replaces null values based on class-conditional mean of
	whether student graduated.'''

	# Group data by graduation
	grad = df.groupby('Graduated')

	# Find means by graduated for each field
	for field in fields:
		mean_no = grad[field].mean()['No']
		mean_yes = grad[field].mean()['Yes']

		# Replace Null Values
		for index, row in df.iterrows():
			if pd.isnull(row[field]):
				if row['Graduated'] == 'No':
					df.ix[index, field] = mean_no
				if row['Graduated'] == 'Yes':
					df.ix[index, field] = mean_yes

	return df


# -------------------------------------------------------------------------
if __name__ == '__main__':
	
	#dataset = 'mock_student_data.csv'
	#df = pd.read_csv(dataset, index_col=0)

	num_fields = ['Age', 'GPA', 'Days_missed']
	non_num_fields = ['State', 'Gender', 'Graduated']

	# Task 1
	#summarize(df, num_fields, non_num_fields, 'task1_summary_stats.txt')

	#for field in num_fields:
	#	histogram(df, field)

	# Task 2
	#genderize(df).to_csv('genderize.csv')

	# Task 3
	dataset2 = 'genderize.csv'
	df2 = pd.read_csv(dataset2, index_col=0)

	#replace_with_mean(df2, num_fields).to_csv('approachA.csv')
	replace_by_graduated(df2, num_fields).to_csv('approachB.csv')
	

