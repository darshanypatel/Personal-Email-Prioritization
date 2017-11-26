% Author: Alper Ender

clc; clear; fclose('all')

%% Reading in the data

% Opening the training file to read
train_fid = fopen('C:\Users\Alper Ender\Downloads\Version 2\Supervised Test\Train.csv','r');

% Initalizing training variables
counter = 1;
train_x = {};
train_y = {};
train_data = {};

% Running through the trining file
while ~feof(train_fid)
    
    % Read the line
    line = fgetl(train_fid);
    
    % Tokenize the line from the delimiter
    tokens = split(line,',');
    
    % Store the emails into the training set
    train_data(counter,:) = tokens(1:end-2);
    train_x(counter,:) = tokens(end-1)';
    train_y(counter,1) = tokens(end);
    
    % Increment counter
    counter = counter + 1;
    
end

fclose(train_fid);


% Opening the test file to read
test_fid = fopen('C:\Users\Alper Ender\Downloads\Version 2\Supervised Test\Testing.csv','r');

% Initalizing test variables
counter = 1;
test_x = {};
test_y = {};
test_data = {};

% Running through the test file
while ~feof(test_fid)
    
    % Read the line
    line = fgetl(test_fid);
    
    % Tokenize the line from the delimiter
    tokens = split(line,',');
    
    % Store the emails into the testing set
    test_data(counter,:) = tokens(1:end-2);
    test_x(counter,:) = tokens(end-1)';
    test_y(counter,1) = tokens(end);
    
    % Increment counter
    counter = counter + 1;
    
end

fclose(test_fid);


%% Supervised Model

% Creating the tokenized documents
train_docs = tokenizedDocument;
test_docs = tokenizedDocument;

% Reading in the dictionary words and tokenizing the words
dictionary = fileread('C:\Users\Alper Ender\Downloads\Version 2\Unsupervised Final\words.txt');
dict = tokenizedDocument(dictionary);

% Creating a bag of words based off the dictionary values
t_bag = bagOfWords(dict)

% - TRAINING -
% Going through the words and creating a tokenized document based on the
% training cleaned words
for i = 1:length(train_x)
    train_docs(i) = tokenizedDocument(train_x{i});
end

% Encode the training words based upon the dictionary words
s_train_x = encode(t_bag, train_docs);


%%
%s_train_x = uint16(s_train_x)
% [r, c] = size(s_train_x);
% b = floor(r/2);
% 
% s1 = full(s_train_x(1:b,:));
% train_x = s1;

% Obtain the entire training set
train_x = full(s_train_x);

%%

% - TESTING -
% Going through the words and creating a tokenized document based on the
% tessting cleaned words
for i = 1:length(test_x)
    test_docs(i) = tokenizedDocument(test_x{i});
end

% Encode the testing words based upon the dictionary words
s_test_x = encode(t_bag, test_docs);

% [r, c] = size(s_test_x);
% b = floor(r/2);
% 
% s2 = full(s_test_x(1:b,:));
% test_x = s2;

% Obtaining the entire testing set
test_x = full(s_test_x);


%% Naive Bayes Supervised Model

% Initializing variables
count = 1;
acc = [];

testing_pca_vals = 5:5:60;

for pca_count = testing_pca_vals
    
    % Calculating the pca for the full trained x matrix
    [coeff,score,latent,tsquared,explained,mu] = pca(train_x, 'NumComponents', pca_count);
    % whos coeff score latent tsquared explained mu
    
    % Finding the mean and the total number of important attributes to keep
    mean_train_x = mean(train_x);
    total_important_values = sum(explained > 1);
    
    % Creating the naive bayes model model
    Mdl = fitcnb(score,train_y);
    
    % Modfiying the testing data using the PCA values
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the values using the testing dataset
    predicted_vals = Mdl.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    num_incorrect = sum(comparison == 0);
    num_correct = sum(comparison == 1);
    
    % Calculating the accuracy for this specific number of pca components
    accuracy = (num_correct / length(comparison)) * 100;
    
    % Printing the values
    fprintf('Num Components: %d - %.2f%% Accuracy\n', pca_count, accuracy);
    
    % Storing the accuracy
    acc(count,1) = accuracy;
    count = count + 1;
    
end

%% Plotting the results

% Creating the figure
figure()

% Creating the scatter plot based upon the results
scatter(testing_pca_vals, acc, '*k')

% Printing the text percentage values
for i = 1:length(acc)
    text(testing_pca_vals(i), acc(i)+3, sprintf('%.f%%', acc(i)) )
end

% Fixing the limits and the grid
axis([0 65 0 100])
grid on

% Labeling Axis
title('Naive Bayes Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories - 50k')
xlabel('Number of PCA Components')
ylabel('Accuracy (Percentage)')


%% Binary Tree Classifier

% Initializing variables
count = 1;
acc = [];

testing_pca_vals = 5:5:60;

for pca_count = testing_pca_vals
    
    % Calculating the pca for the full trained x matrix
    [coeff,score,latent,tsquared,explained,mu] = pca(train_x, 'NumComponents', pca_count);
    % whos coeff score latent tsquared explained mu
    
    % Calculating the mean and the total number of important attributes to keep
    mean_train_x = mean(train_x);
    total_important_values = sum(explained > 1);
    
    % Creating a binary tree classifier model
    Mdl = fitctree(score, train_y);
    
    % Reshaping the testing dataset based upon the pca coefficients
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the modified values
    predicted_vals = Mdl.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    num_incorrect = sum(comparison == 0);
    num_correct = sum(comparison == 1);
    
    % Calculating the accuracy for this specific number of pca components
    accuracy = (num_correct / length(comparison)) * 100;
    
    % Printing the values
    fprintf('Num Components: %d - %.2f%% Accuracy\n', pca_count, accuracy);
    
    % Storing the accuracy
    acc(count,1) = accuracy;
    count = count + 1;
    
end

%% Plotting the results

% Creating the figure
figure()

% Creating the scatter plot based upon the results
scatter(testing_pca_vals, acc, '*k')

% Printing the text percentage values
for i = 1:length(acc)
    text(testing_pca_vals(i), acc(i)+3, sprintf('%.f%%', acc(i)) )
end

% Fixing the limits and the grid
axis([0 65 0 100])
grid on

% Labeling the axis
title('Binary Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories - 50k')
xlabel('Number of PCA Components')
ylabel('Accuracy (Percentage)')

%% KNN Classifier

% Initializing variables
count = 1;
acc = [];

testing_pca_vals = 5:5:60;

for pca_count = testing_pca_vals
    
    % Calculating the pca for the full trained x matrix
    [coeff,score,latent,tsquared,explained,mu] = pca(train_x, 'NumComponents', pca_count);
    % whos coeff score latent tsquared explained mu
    
    % Calculating the mean and the total number of important attributes to keep
    mean_train_x = mean(train_x);
    total_important_values = sum(explained > 1);
    
    % Creating a KNN classifier model
    Mdl = fitcknn(score, train_y);
    
    % Reshaping the testing dataset based upon the pca coefficients
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the modified values
    predicted_vals = Mdl.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    num_incorrect = sum(comparison == 0);
    num_correct = sum(comparison == 1);
    
    % Calculating the accuracy for this specific number of pca components
    accuracy = (num_correct / length(comparison)) * 100;
    
    % Printing the values
    fprintf('Num Components: %d - %.2f%% Accuracy\n', pca_count, accuracy);
    
    % Storing the accuracy
    acc(count,1) = accuracy;
    count = count + 1;
    
end

%% Plotting the results

% Creating the figure
figure()

% Creating the scatter plot based upon the results
scatter(testing_pca_vals, acc, '*k')

% Printing the text percentage values
for i = 1:length(acc)
    text(testing_pca_vals(i), acc(i)+3, sprintf('%.f%%', acc(i)) )
end

% Fixing the limits and the grid
axis([0 65 0 100])
grid on

% Labeling the axis
title('KNN Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories - 50k')
xlabel('Number of PCA Components')
ylabel('Accuracy (Percentage)')


%% NO PCA

% Creating a KNN classifier model
Mdl = fitcknn(train_x, train_y);

% Predicting the modified values
predicted_vals = Mdl.predict(test_x);

% Calculating the number correct and incorrect classifications
comparison = str2double(test_y) == str2double(predicted_vals);
num_incorrect = sum(comparison == 0);
num_correct = sum(comparison == 1);

% Calculating the accuracy for this specific number of pca components
accuracy = (num_correct / length(comparison)) * 100;

% Printing the values
fprintf('NO PCA : %.2f%% Accuracy\n', accuracy);


%% Plotting the results

% Creating the figure
figure()

% Obtaining the number of rows and columns
[r,c] = size(test_x);

% Creating the scatter plot based upon the results
scatter(c,accuracy, '*k')

% Printing the text percentage values
for i = 1:length(c)
    text(c(i), accuracy+3, sprintf('%.2f%%', accuracy) )
end

% Fixing the limits and the grid
ylim([0 100])
grid on

% Labeling the axis
title('KNN Classifier - Accuracy vs. NO PCA - 20 Categories - 50k')
xlabel('Number of Components')
ylabel('Accuracy (Percentage)')



%% Ensemble of Learners for classification tree

% Initializing variables
count = 1;
acc = [];

testing_pca_vals = 10:10:60;

for pca_count = testing_pca_vals
    
    % Calculating the pca for the full trained x matrix
    [coeff,score,latent,tsquared,explained,mu] = pca(train_x, 'NumComponents', pca_count);
    % whos coeff score latent tsquared explained mu
    
    % Calculating the mean and the total number of important attributes to keep
    mean_train_x = mean(train_x);
    total_important_values = sum(explained > 1);
    
    % Creating an ensemble model with multiple parameters
    templ = templateTree('Surrogate','all');
    ens = fitcensemble(score, train_y,'OptimizeHyperparameters','auto');
    
    % Reshaping the testing dataset based upon the pca coefficients
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the modified values
    predicted_vals = ens.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    num_incorrect = sum(comparison == 0);
    num_correct = sum(comparison == 1);
    
    % Calculating the accuracy for this specific number of pca components
    accuracy = (num_correct / length(comparison)) * 100;
    
    % Printing the values
    fprintf('Num Components: %d - %.2f%% Accuracy\n', pca_count, accuracy);
    
    % Storing the accuracy
    acc(count,1) = accuracy;
    count = count + 1;
    
end

%% Plotting the results

% Creating the figure
figure()

% Creating the scatter plot based upon the results
scatter(testing_pca_vals, acc, '*k')

% Printing the text percentage values
for i = 1:length(acc)
    text(testing_pca_vals(i), acc(i)+3, sprintf('%.f%%', acc(i)) )
end

% Fixing the limits and the grid
axis([0 65 0 100])
grid on

% Labeling the axis
title('Ensemble Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories - 50k')
xlabel('Number of PCA Components')
ylabel('Accuracy (Percentage)')



