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
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Code adapted from https://github.com/yhat/DataGotham2013/

class MLPipeline():
	"""
	MLPipeline takes testing and training data and a list of methods. It returns predicted values in a CSV.

	To run directly in command line:
	python pipeline.py [training_data] [testing_data] [methods] [run_immediately]

	Method options include:
	* 'linear_reg': linear regression

	"""

	def __init__(self, training_data = None, testing_data = None, run_immediately = True):
		
		if training_data:
			self.training_data = training_data
		if testing_data:
			self.testing_data = testing_data
		
		if run_immediately:
			# Load Training Data
			self.df = self.read_data(self.training_data)

			# Output Training Summary Statistics
			self.summary = self.summarize(self.df)

			# Load Testing Data
			self.test_df = self.read_data(self.testing_data)


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

	def summarize(self, df):
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
		return output

	def print_to_csv(self, df, filename):
		''' Given a dataframe and a filename, outputs a CSV.'''

		# Check that filename was CSV
		filename_split = str(filename).strip().split('.')

		if filename_split[-1] != 'csv':
			filename = "".join(filename_split[0:len(filename_split)-1]) + ".csv"

		df.to_csv(filename)

	def histogram(df, field, bins):
		'''Given a dataframe, a field and a bin count, this creates histograms.'''

		fn = field + '.png'

		#Create histogram
		df[field].hist(xlabelsize=10, ylabelsize=10, bins=bins)
		pylab.title("Distribution of {0}".format(field))
		pylab.xlabel(field)
		pylab.ylabel("Counts")
		pylab.savefig(fn)
		pylab.close()

# ---------------------------------------------------------------------

	## 3. Pre-Process Data: Fill in misssing values

	def replace_with_value(self, df, variables, values):
		'''
		For some variables, we can infer what the missing value should be.
		This function takes a dataframe and a list of variables that match
		this criteria and replaces null values with the specified value.
		'''
		for i in range(len(variables)):
			variable = variables[i]
			value = values[i]
			df[variable] = df[variable].fillna(value)


	def impute(self, df, variable, cols):
		'''
		For some variables, we cannot infer the missing value and replacing with
		a conditional mean does not make sense.
		This function takes a dataframe and a variables that matches this criteria 
		as well as a list of columns to calibrate with and uses nearest neighbors 
		to impute the null values.
		'''
		# Split data into test and train for cross validation
		is_test = np.random.uniform(0, 1, len(df)) > 0.75
		train = df[is_test==False]
		validate = df[is_test==True]
		
		## Calibrate imputation with training data
		imputer = KNeighborsRegressor(n_neighbors=1)

		# Split data into null and not null for given variable
		train_not_null = train[train[variable].isnull()==False]
		train_null = train[train[variable].isnull()==True]

		# Replace missing values
		imputer.fit(train_not_null[cols], train_not_null[variable])
		new_values = imputer.predict(train_null[cols])
		train_null[variable] = new_values

		# Combine Training Data Back Together
		train = train_not_null.append(train_null)

		# Apply Nearest Neighbors to Validation Data
		new_var_name = variable + 'Imputed'
		validate[new_var_name] = imputer.predict(validate[cols])
		validate[variable] = np.where(validate[variable].isnull(), validate[new_var_name],
									validate[variable])


		# Drop Imputation Column & Combine Test & Validation
		validate.drop(new_var_name, axis=1, inplace=True)
		df = train.append(validate)

		''' FOR FUTURE
		# Apply Imputation to Testing Data
		test_df[new_var_name] = imputer.predict(test_df[cols])
		test_df[variable] = np.where(test_df[variable].isnull(), test_df[new_var_name],
									test_df[variable])
		'''

		return df.sort_index()
		

# ---------------------------------------------------------------------

	## 4. Generate Features: Write a sample function that can discretize a continuous variable
	## and one function that can take a categorical variable and create binary variables from it.

	def find_features(self, df, features, variable):
		'''Uses random forest algorithm to determine the relative importance of each
		potential feature. Takes dataframe, a numpy array of features, and the dependent
		variable. Outputs dataframe, sorting features by importance'''
		clf = RandomForestClassifier()
		clf.fit(df[features], df[variable])
		importances = clf.feature_importances_
		sort_importances = np.argsort(importances)
		rv = pd.DataFrame(data={'variable':features[sort_importances],
								'importance':importances[sort_importances]})
		return rv

	def adjust_outliers(self, x, cap):
		'''Takes series and creates upperbound cap to adjust for outliers'''
		if x > cap:
			return cap
		else:
			return x

	def bin_variable(self, df, variable, num_bins, labels=None):
		'''Discretizes a continuous variable based on specified number of bins'''
		new_label = variable + '_bins'
		df[new_label] = pd.cut(df[variable], bins=num_bins, labels=labels)

	def get_dummys(self, df, cols):
		'''Creates binary variable from specified column(s)'''
		# Loop through each variable
		for variable in cols:
			dummy_data = pd.get_dummies(df[variable], prefix=variable)
			
			# Add new data to dataframe
			df = pd.concat([df, dummy_data], axis=1)
	

# ---------------------------------------------------------------------

	## 5. Build Classifier: For this assignment, select any classifer you feel comfortable with 
	## (Logistic Regression for example)
	## &&
	## 6. Evaluate Classifier: you can use any metric you choose for this assignment (accuracy 
	## is the easiest one)

	def nearest_neighbors(self, df, features, variable):
		'''Uses nearest neighbors to generate and evaluate predictions of a specified
		variable.'''
		
		# Split data into test and train for cross validation
		is_test = np.random.uniform(0, 1, len(df)) > 0.75
		train = df[is_test==False]
		validate = df[is_test==True]
		
		# Run Nearest Neighbors and Validate
		clf = KNeighborsClassifier()
		clf.fit(train[features], train[variable])
		validate['predictions'] = clf.predict(validate[features])

		# Create CSV of Predictions
		validate.to_csv('predictions.csv')
		
		return accuracy_score(validate[variable], validate['predictions'])
	


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
		


