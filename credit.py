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
import pipeline as ml 


def credit_model(dataset):
	
	#######################################################
	# Load Credit Data and Run Initial Summary Statistics #
	#######################################################
	print "Loading data..."

	## LOAD DATA
	df = ml.read_data(dataset)
	variables = list(df.columns.values)

	## RUN INITIAL SUMMARY STATISTICS & GRAPH DISTRIBUTIONS
	summary = ml.summarize(df)
	#print_to_csv(summary, 'summary_stats.csv')
	
	for v in variables:
		ml.histogram(df, v)

	## FOR FUTURE: Drop rows where 'percentage' fields have values > 1

	############################
	# Deal with missing values #
	############################
	print "Handling missing values..."

	print "Correcting dependents column..."
	''' DEPENDENTS: Missing values are likely zeros. If someone didn't 
	provide this info, they likely wouldn't have kids.'''
	variables = ['NumberOfDependents']
	values = [0]
	ml.replace_with_value(df, variables, values)

	print "Correcting income column..."
	'''MONTHLY INCOME: It wouldn't make sense to determine missing values
	through replacing with a specific value. Instead, impute null values with
	the mean of income.'''
	variables = ['MonthlyIncome']
	ml.replace_with_mean(df, variables)

	#ml.print_to_csv(df, 'credit-data-updated.csv')

	#####################
	# Generate Features #
	#####################
	print "Generating features..."

	## FIND IMPORTANT FEATURES
	test_features = np.array(['RevolvingUtilizationOfUnsecuredLines', 'age',
							'NumberOfTime30-59DaysPastDueNotWorse',
							'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
							'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
							'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'])
	y = 'SeriousDlqin2yrs'

	## Find initial best features
	#print ml.find_features(df, test_features, y)

	## ENGINEER ADDITIONAL FEATURES
	print "Engineering income buckets..."
	'''MONTHLY INCOME: Break this into buckets, adjusting for outliers'''
	df['MonthlyIncome_adjust'] = df.MonthlyIncome.apply(lambda x: ml.adjust_outliers(x, 15000))	
	ml.bin_variable(df, 'MonthlyIncome_adjust', 15, False)
	#print pd.value_counts(df['MonthlyIncome_adjust_bins'])

	print "Engineering age buckets..."
	'''AGE: Break this into buckets'''
	bins = [-1] + range(20, 80, 5) + [120]
	ml.bin_variable(df, 'age', bins, False)
	#print pd.value_counts(df['age_bins'])
	
	#print df.head()

	## RECALCULATE IMPORTANT FEATURES
	new_features = np.array(['MonthlyIncome_adjust_bins', 'age_bins'])
	all_features = np.hstack((test_features, new_features))
	#print all_features
	#print ml.summarize(df)
	#ml.print_to_csv(df, 'test.csv')

	## FIND BEST FEATURES
	#print ml.find_features(df, all_features, y)

	### FOR FUTURE: It would be cool to be able to automatically point to the top
	### five best features or focus on the features that meet a certain threshold
	

	################################
	# Build & Evaluate Classifiers #
	################################
	print "Evaluating classifiers..."

	## USE TOP FEATURES TO COMPARE CLASSIFIER PERFORMACE
	X = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio',
					'MonthlyIncome', 'age', 'NumberOfTimes90DaysLate',
					'NumberOfOpenCreditLinesAndLoans']

	print ml.evaluate_classifiers(df, X, y)

# -------------------------------------------------------------------------
if __name__ == '__main__':
	
	dataset = "data/cs-training.csv"
	credit_model(dataset)
