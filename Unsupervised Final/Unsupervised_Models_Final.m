%% Notes

%{

Important Links (SSE):
    http://www.cs.uky.edu/~jzhang/CS689/PPDM-Chapter3.pdf
    https://www.cse.buffalo.edu/~jing/cse601/fa12/materials/clustering_basics.pdf
    https://hlab.stanford.edu/brian/error_sum_of_squares.html

%}

%% Clearing Everything

clc; clear; close all;

%% Setup

% Location of the CSV File or Folder System
FOLDER_LOCATION = '/Users/alperender/Desktop/ALDA-Project/Sampling/';  % (Mac)
% FOLDER_LOCATION = 'C:\Users\Alper Ender\Desktop\inbox';       % (Windows)

% The Filename to read in
FILENAME = 'CleanBody_Samples_881.csv';

%% Read in English Dictionary

% words.txt taken from:
% https://raw.githubusercontent.com/dwyl/english-words/master/words.txt

% Open the file to read from
FID = fopen('words.txt','r');

% Setup variables
words = {};
i = 1;

while ~feof(FID)
    
    % Read in words from the file
    words{i,1} = fgetl(FID);
    i = i+1;
    
end

fclose('all');

%% Read in the values

% PreProcess referenced and used from:
% https://www.mathworks.com/help/textanalytics/
%         examples/prepare-text-data-for-analysis.html

FID = fopen([FOLDER_LOCATION FILENAME],'r');

% Initializing docs and count
docs = tokenizedDocument;
count = 1;

num = regexp(FILENAME,'([0-9]*)','tokens');
value = str2double(num{1}{1});
date  = cell(1,1);
from  = cell(1,1);
to    = cell(1,1);
subj  = cell(1,1);
loc   = cell(1,1);


% Iterating through number of files
while ~feof(FID)
    
    % Reading the line
    line = fgetl(FID);
    
    % Split based on commas
    vals = split(line, ',');
    
    % Obtain the values from the sampled data
    date{count,1} = vals{1};
    from{count,1} = vals{2};
    to{count,1}   = vals{3};
    subj{count,1} = vals{4};
    loc{count,1}  = vals{5};
    
    % Preprocess the values
    docs(count,1) = PreProcess([vals{4} vals{6}]);
    
    % Increment the counter
    
    if mod(count,1000)==0
        disp(count)
    end
    count = count+1;
    
end

% Close the file
fclose(FID);


%% Batch PreProcessing

% Preprocessing all the documents in a batch
docs = BatchPreProcess(docs);

% Go through the vocabulary and delete all non-dictionary words
document_vocabulary = docs.Vocabulary;

% Go through each word in the vocabulary and checking to see if it is in
% the list and delete the word from the document if it is not
del_words = {};
count = 1;
for i = 1:length(document_vocabulary)
    if sum( strcmpi(document_vocabulary{i} , words)) == 0
        docs = docs.removeWords(document_vocabulary{i});
    end
    if mod(i,1000) == 0
        fprintf('%d out of %d\n',i,length(document_vocabulary))
    end
end

% docs = docs.normalizeWords();

%% Bag of Words Model

% Creating the bag of words of model
bag = bagOfWords(docs);

% Removing infrequent words
bag = removeInfrequentWords(bag,2);

% Obtaining the Number of Words and the Number of Documents
numberOfWords = bag.NumWords;
numberOfDocuments = bag.NumDocuments;


%% TFIDF Model

% Creating the tfidf model
M = tfidf(bag);

% Looking at the model
m = full(M);
topkwords(bag,10)

% Getting the size of M
size(M)
full_M = full(M);


%% TFIDF, LDA, K-Medoids

numTopics = 20;
mdl = fitlda(round(full_M), numTopics)
% mdl = fitlda(bag, numTopics)

% Setting up figure
figure()
set(gcf,'Name', 'TFIDF, LDA, K-Medoids')

% Grouping the documents based on k-medoids
[groups,C] = kmedoids(mdl.DocumentTopicProbabilities, numTopics,...
    'Distance'  , 'sqeuclidean');

% Calculating PCA for 2 values only to visualize the plot
[coeff,score] = pca(mdl.DocumentTopicProbabilities,'NumComponents',2);

% Plotting Values
colors = linspecer();
scatter(score(:,1), score(:,2), [], colors(groups)');

% Title
title(['K-Means sqeuclidean'])
grid on

%% Export Values

% Opening the file to write the cleaned data to
um_fid = fopen('Unsupervised Output.csv','w');

% Iterating through the documents
for i = 1:length(docs)
    
    % Clearning an array to put the strings into
    words = [];
    word_vals = docs(i).string';
    
    % Go through the cleaned words and put them into the 
    for j = 1:length(word_vals)
        words = [words ' ' char(word_vals(j))];
    end
    
    % Printing to file
    fprintf(um_fid, '%s,%s,%s,%s,%s,%s,%d\n', date{i}, from{i}, to{i}, subj{i}, loc{i}, words, groups(i));
    
end

% Closing file
fclose(um_fid);

