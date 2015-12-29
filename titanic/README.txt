TITANIC: Machine Learning from Disaster

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this kaggle challenge, analysis is conducted to investigate what sorts of people were likely to survive. In particular, the tools of machine learning are applied to predict which passengers survived the tragedy.
 
Directory Contents:

	- test.cv, train.csv
	Files that contain the raw data for the task, training data is used to 	optimise the model so that we can make accurate predictions on the test 	set.	

	- titantic_ML.py
	A python script that uses a number of machine learning specific libraries 
	to munge the data and fit a model that predicts the survival of passengers
	based data in the test set.

	The training & test sets contain missing values so there are a number of
	different approaches used to fill in the blanks, in some cases where there
	are only a few missing values we use an average, for more intricate data
	a general linear model is used. 

	Two machine learning methods are evaluated for statistical inference, a 
	logistic classifier and a support vector machine. The optimum parameters 
	for these models are evaluated in the script and selected automatically
	using the a segment of the training data for cross validation.

	The script saves the results from the highest performing model and can be
	found in the working directory once the script has run.

	- RegTools.py
	A collection of python functions that are useful for preparing data and 
	implementing regression models. A function is used in the main script
	to normalise the data for both the training and test datasets.
	