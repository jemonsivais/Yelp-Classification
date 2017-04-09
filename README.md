# Yelp-Classification
Python script for the Yelp dataset challenge.
This script asumes the json data has been converted into a .csv file using the scripts provided by Yelp.
The script uses half the yelp data to train the algorithm and the other half to test it.

## Algorithm
The algorithm used is Naive bayes classification for documents. 
The classification space is the number of stars a review can receive (1, 2, 3, 4, 5) 
and each classification has a probability of having a word in the dictionary, which is taken from the training section.

## Results
The algorithm will print the correct predictions and the incorrect predictions, it will also provide
the error distance which is a metric of how far the predicted classification was from the actual classification 
calculated as the average of the absoulute difference between the actual classification and predicted classification.
