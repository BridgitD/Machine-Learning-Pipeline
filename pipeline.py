##################################
##								##
##	Machine Learning Pipeline 	##
##	Bridgit Donnelly			##
##								##
##################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys
import sklearn as sk

# Code adapted from Chapin Hall python library & https://github.com/yhat/DataGotham2013/

class MLPipeline():
	"""
	MLPipeline takes testing and training data and a list of methods. It returns predicted values in a CSV.

	To run directly in command line:
	python pipeline.py [training_data] [testing_data] [methods] [run_immediately]

	Method options include:
	* 'linear_reg': linear regression

	"""

	def __init__(self, training_data = None, testing_data = None, methods = [], run_immediately = True):
		
		if training_data:
			self.training_data = training_data
		if testing_data:
			self.testing_data = testing_data
		if methods:
			self.methods = methods

		if run_immediately:
			# Load Training Data
			self.df = self.read_data(self.training_data)

			# Output Training Summary Statistics
			self.summary = self.summarize(self.df, self.training_data)


	## 1. Read Data: For this assignment, assume input is CSV

	def read_data(self, filename):
		'''
		Takes a filename and returns a dataframe.
		'''
		original = pd.read_csv(filename, index_col=0)
		df = original.copy()
		return df

# ---------------------------------------------------------------------

	## 2. Explore Data: You can use the code you wrote for assignment 1 here to generate 
	## distributions and data summaries

	def summarize(self, df, filename):
		'''
		Given a dataframe and the original filename, this outputs a CSV with an adjusted filename.
		Return the summary table.
		'''

		# Create Summary Table
		summary = df.describe().T

		# Add Median & Missing Value Count
		summary['median'] = df.median()
		summary['missing_vals'] = df.count().max() - df.describe().T['count'] # Tot. Count - count
		
		output = summary.T

		# Adjust filename
		filename_split = str(filename).strip().split('.')
		output_fn = "".join(filename_split[0:len(filename_split)-1]) + "_summary.csv"

		output.to_csv(output_fn)

		return output

# ---------------------------------------------------------------------

	## 3. Pre-Process Data: Fill in misssing values

	def replace_with_value(self, df, variables, values):
		'''
		For some variables, we can infer what the missing value should be.
		This function takes a dataframe and a list of variables that matches
		this criteria and replaces null values with the specified value.
		'''
		for i in range(len(variables)):
			variable = variables[i]
			value = values[i]
			df[variable] = df[variable].fillna(value)



# ---------------------------------------------------------------------

	## 4. Generate Features: Write a sample function that can discretize a continuous variable
	## and one function that can take a categorical variable and create binary variables from it.


# ---------------------------------------------------------------------

	## 5. Build Classifier: For this assignment, select any classifer you feel comfortable with 
	## (Logistic Regression for example)
	## &&
	## 6. Evaluate Classifier: you can use any metric you choose for this assignment (accuracy 
	## is the easiest one)


	def run_methods(methods):
		'''
		Takes a list of methods and outputs CSV file for each with predicted values
		'''


	


# -------------------------------------------------------------------------
if __name__ == '__main__':
	
	if len(sys.argv) <= 1:
		sys.exit("Must include a filename.")

	else:
		training_data = sys.argv[1]
		testing_data = sys.argv[2]

		MLPipeline(training_data, testing_data)

		'''
		# Load Data
		df = read_data(training_data)

		# Produce Summary Stats
		output_fn = training_data + "_summary-stats.csv"
		summarize(df, output_fn)
		'''
		


