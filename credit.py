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
	
	methods = ['linear_reg']

	#######################################################
	# Load Credit Data and Run Initial Summary Statistics #
	#######################################################
	
	credit = ml.MLPipeline(training_data, testing_data, methods)
	#print credit.summary
	###credit.print_to_csv(credit.summary, 'summary_stats.csv')
	#print pd.value_counts(credit.df.MonthlyIncome, sort=False)

	############################
	# Deal with missing values #
	############################

	''' Dependents: Missing values are likely zeros. If someone didn't 
	provide this info, they likely wouldn't have kids.'''
	variables = ['NumberOfDependents']
	values = [0]
	credit.replace_with_value(credit.df, variables, values)
	#print pd.value_counts(credit.df.NumberOfDependents)
	
	'''Monthly Income: It wouldn't make sense to determine missing values
	through replacing with a specific value. Instead, use the NearestNeighbors
	algorithm to impute the null values based on variables with highest correlation
	to monthly income.'''
	
	variable = 'MonthlyIncome'
	
	# For non-null instances of monthly income, calculate correlations
	non_null = credit.df[credit.df.MonthlyIncome.isnull()==False]
	#print non_null.corr().ix[:,5]
	# Top 5 correlated variables
	calibration_variables = ['NumberOfOpenCreditLinesAndLoans', 
							'NumberRealEstateLoansOrLines',
							'NumberOfDependents',
							'age',
							'DebtRatio']
	
	# Take out all values greater than 3 standard deviations
	#no_outliers = credit.df[np.abs(credit.df.MonthlyIncome - credit.df.MonthlyIncome.mean() 
					<= (3*credit.df.MonthlyIncome.std()))]
	
	
	# Run Imputation
	train, test = credit.impute(credit.df, variable, calibration_variables)

	credit.print_to_csv(train, 'credit-train-data.csv')
	credit.print_to_csv(test, 'credit-test-data.csv')
	


# -------------------------------------------------------------------------
if __name__ == '__main__':
	
	training_data = "credit_data/cs-training.csv"
	testing_data = "credit_data/cs-test.csv"

	credit_model(training_data, testing_data)
