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


def credit_model(training_data, testing_data):
	
	#######################################################
	# Load Credit Data and Run Initial Summary Statistics #
	#######################################################
	
	credit = ml.MLPipeline(training_data, testing_data)
	#print credit.summary
	credit.print_to_csv(credit.summary, 'summary_stats.csv')
	#print pd.value_counts(credit.df.MonthlyIncome, sort=False)

	############################
	# Deal with missing values #
	############################

	''' DEPENDENTS: Missing values are likely zeros. If someone didn't 
	provide this info, they likely wouldn't have kids.'''
	variables = ['NumberOfDependents']
	values = [0]
	credit.replace_with_value(credit.df, variables, values)
	#print pd.value_counts(credit.df.NumberOfDependents)
	
	'''MONTHLY INCOME: It wouldn't make sense to determine missing values
	through replacing with a specific value. Instead, use the NearestNeighbors
	algorithm to impute the null values based on variables with highest correlation
	to monthly income.'''
	
	variable = 'MonthlyIncome'
	
	## FOR NON-NULL INSTANCES OF MONTHLY INCOME, CALCULATE CORRELATIONS
	non_null = credit.df[credit.df.MonthlyIncome.isnull()==False]
	#print non_null.corr().ix[:,5]

	## TOP 5 CORRELATED VARIABLES
	calibration_variables = ['NumberOfOpenCreditLinesAndLoans', 
							'NumberRealEstateLoansOrLines',
							'NumberOfDependents',
							'age',
							'DebtRatio']
	
	## FOR FUTURE: Take out all values greater than 3 standard deviations - not working
	#no_outliers = credit.df[np.abs(credit.df.MonthlyIncome - credit.df.MonthlyIncome.mean() 
	#				<= (3*credit.df.MonthlyIncome.std()))]
	
	## RUN IMPUTATION FOR MONTHLY INCOME
	credit.df = credit.impute(credit.df, variable, calibration_variables)
	#print pd.value_counts(credit.df.MonthlyIncome)
	
	credit.print_to_csv(credit.df, 'credit-train-data-updated.csv')
	
	#####################
	# Generate Features #
	#####################

	## FIND IMPORTANT FEATURES
	features = np.array(['RevolvingUtilizationOfUnsecuredLines', 'age',
							'NumberOfTime30-59DaysPastDueNotWorse',
							'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
							'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
							'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'])
	variable = 'SeriousDlqin2yrs'

	#print credit.find_features(credit.df, features, variable)

	## ENGINEER ADDITIONAL FEATURES
	'''MONTHLY INCOME: Break this into buckets, adjusting for outliers'''
	credit.df['MonthlyIncome_adjust'] = credit.df.MonthlyIncome.apply(lambda x: credit.adjust_outliers(x, 15000))	
	credit.bin_variable(credit.df, 'MonthlyIncome_adjust', 15, False)
	#print pd.value_counts(credit.df['MonthlyIncome_adjust_bins'])

	'''AGE: Break this into buckets'''
	bins = [0] + range(20, 80, 5) + [120]
	credit.bin_variable(credit.df, 'age', bins, False)
	#print pd.value_counts(credit.df['age_bins'])
	#print credit.df.head()

	## RECALCULATE IMPORTANT FEATURES
	new_features = np.array(['MonthlyIncome_adjust_bins', 'age_bins'])
	all_features = np.hstack((features, new_features))
	#print all_features
	#print credit.df.head()
	#print credit.find_features(credit.df, all_features, variable)

	### FOR FUTURE: It would be cool to be able to automatically point to the top
	### five best features or focus on the features that meet a certain threshold
	

	###############################
	# Build & Evaluate Classifier #
	###############################

	## USE TOP FEATURES TO CALCULATE NEAREST NEIGHBORS
	top_features = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio',
					'MonthlyIncome', 'age', 'NumberOfTimes90DaysLate',
					'NumberOfOpenCreditLinesAndLoans']

	# Output CSV of Test Predictions & Accuracy Score
	print "Accuracy: " + str(credit.nearest_neighbors(credit.df, top_features, variable))




# -------------------------------------------------------------------------
if __name__ == '__main__':
	
	training_data = "data/cs-training.csv"
	testing_data = "data/cs-test.csv"

	credit_model(training_data, testing_data)
