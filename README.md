# CSC522_Project
Personal Email Prioritization

Authors:
* Alper E.
* Darshan P.
* Rohit N.
* Tushita R.

# Required Software Packages
* MATLAB - Version 9.3 - (R2017b)
* Statistics and Machine Learning Toolbox - Version 11.2 - (R2017b)
* Text Analytics Toolbox - Version 1.0 - (R2017b)

# Description of folders
## Archive
Holds MATLAB files to convert the folder system of the email corpus to a csv file. Also includes testing files.

## IndividualFiles
Holds the individual .csv files for each Enron email account.

## RawData
Holds the large uncleaned .csv file that was extracted from the folder system of emails.

## Sampling
Holds the MATLAB files to sample and clean the emails. As well, holds the various sampled .csv datasets.

## SocialNetwork
Holds the MATLAB file to create the social network and calculate importance of each email according to the importance formula.

## SupervisedFinal
Holds the MATLAB files to create the final supervised model.

## SupervisedTest
Holds the MATLAB files to test the supervised models. In addition, holds the .png and .txt files to visualize results.

## UnsupervisedFinal
Holds the MATLAB files to create the final unsupervised model.

## UnsupervisedTest
Holds the MATLAB files to test the unsupervised models. In addition, holds the .png files to visualize results.

# Dataset
The Enron email dataset was taken from the Carnegie Mellon University Corpus located here: https://www.cs.cmu.edu/~enron/

# Note
Please note that this code contains the files to calculate the importance of an email. It does not create an application or simulate actually taking in a new email. The purpose is for soley testing the algorithms and methods described in our paper.

# Code References
* DBSCAN - https://www.mathworks.com/matlabcentral/fileexchange/52905-dbscan-clustering-algorithm
* Dictionary text file - https://raw.githubusercontent.com/dwyl/english-words/master/words.txt
* Preprocessing Tokenized Documents - https://www.mathworks.com/help/textanalytics/examples/prepare-text-data-for-analysis.html