%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% @AutoGenerated
%
% Filename: Supervised_Model_Test.m
% Author: Alper Ender
% Date: November 2017
% Description: Testing the supervised models. Testing against the type of
% model and the number of dimensions to use
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\

clc; fclose('all');

cd(folder_system.SupervisedTest)

% Percent of the values that are in the training data
PERCENT_TRAIN = 80;

% Opening the file to read
FID = fopen('Unsupervised Output.csv','r');


%% Importing values

fprintf('Beginning Supervised Model Test...\n')
fprintf('Importing Documents...\n')

% Initalizing variables
counter = 1;
all_emails = {};

% Running through the file
while ~feof(FID)
    
    % Read the line
    line = fgetl(FID);
    
    % Tokenize the line from the delimiter
    tokens = split(line,',');
    
    % Store the eamil
    all_emails(counter,:) = tokens';
    
    % Increment counter
    counter = counter + 1;
    
end

%% Creating the Bag of Words from the dictionary

fprintf('Creating the Dictionary Bag of Words...\n')

% Reading in the dictionary words and tokenizing the words
dictionary = fileread('words.txt');
dict = tokenizedDocument(dictionary);

% Creating a bag of words based off the dictionary values
t_bag = bagOfWords(dict);


%% Creating the documents Bag of Words

fprintf('Creating the Documents Bag of Words...\n')

% Initializing variables
docs = tokenizedDocument;
[r, c] = size(all_emails);

% Go through each email and put into the document list
for i = 1:r
    docs(i,1) = tokenizedDocument(all_emails{i,end-1});
end

% Encode the documents based off the dictionary bag of words
docs_s = encode(t_bag, docs);
all_docs = full(docs_s);

% Obtaining the split val
split_val = floor(r * 0.8);

% Obtaining the training values
train_x = all_docs(1:split_val, :);
train_y = all_emails(1:split_val, end);

% Obtaining the testing values
test_x = all_docs(split_val + 1 : end, :);
test_y = all_emails(split_val + 1 : end, end);


%% Naive Bayes Supervised Model

fprintf('Creating the Naive Bayes Model...\n')

% Initializing variables
count = 1;
acc = [];

testing_pca_vals = 5:5:60;

try
    
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
        fprintf('Naive Bayes. Num Components: %d - %.6f%% Accuracy\n', pca_count, accuracy);
        
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
    title('Naive Bayes Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories')
    xlabel('Number of PCA Components')
    ylabel('Accuracy (Percentage)')
    
catch
    
    fprintf('Error on Naive Bayes Classifer...\n')
    
end


%% Binary Tree Classifier

fprintf('Creating the Binary Classifier Model...\n')

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
    fprintf('Binary Classifier. Num Components: %d - %.6f%% Accuracy\n', pca_count, accuracy);
    
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
title('Binary Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories')
xlabel('Number of PCA Components')
ylabel('Accuracy (Percentage)')

%% KNN Classifier

fprintf('Creating the KNN Model...\n')

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
    fprintf('KNN. Num Components: %d - %.6f%% Accuracy\n', pca_count, accuracy);
    
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
title('KNN Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories')
xlabel('Number of PCA Components')
ylabel('Accuracy (Percentage)')


%% NO PCA

fprintf('Creating the KNN No PCA Model...\n')

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
fprintf('KNN. NO PCA - %.2f%% Accuracy\n', accuracy);


%% Plotting the results

% Creating the figure
figure()

% Obtaining the number of rows and columns
[r,c] = size(test_x);

% Creating the scatter plot based upon the results
scatter(c,accuracy, '*k')

% Printing the text percentage values
for i = 1:length(c)
    text(c(i), accuracy+3, sprintf('%.6f%%', accuracy) )
end

% Fixing the limits and the grid
ylim([0 100])
grid on

% Labeling the axis
title('KNN Classifier - Accuracy vs. NO PCA - 20 Categories')
xlabel('Number of Components')
ylabel('Accuracy (Percentage)')



%% Optimized Classifiers

%% Naive Bayes Optimized Supervised Model

fprintf('Creating the Optimized Naive Bayes Model...\n')

% Initializing variables
count = 1;
acc = [];

testing_pca_vals = 5:5:60;

try
    
    for pca_count = testing_pca_vals
        
        % Calculating the pca for the full trained x matrix
        [coeff,score,latent,tsquared,explained,mu] = pca(train_x, 'NumComponents', pca_count);
        % whos coeff score latent tsquared explained mu
        
        % Finding the mean and the total number of important attributes to keep
        mean_train_x = mean(train_x);
        total_important_values = sum(explained > 1);
        
        % Creating the naive bayes model model
        Mdl = fitcnb(score,train_y, 'OptimizeHyperParameters', 'auto');
        
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
        fprintf('Optimizied Naive Bayes. Num Components: %d - %.6f%% Accuracy\n', pca_count, accuracy);
        
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
    title('Optimized Naive Bayes Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories')
    xlabel('Number of PCA Components')
    ylabel('Accuracy (Percentage)')
    
catch
    
    fprintf('Error on Naive Bayes Classifer...\n')
    
end


%% Optimized Binary Classifier

fprintf('Creating the Optimized Binary Classifier Model...\n')

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
    Mdl = fitctree(score, train_y, 'OptimizeHyperParameters', 'auto');
    
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
    fprintf('Optimized Binary Classifier. Num Components: %d - %.6f%% Accuracy\n', pca_count, accuracy);
    
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
title('Optmized Binary Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories')
xlabel('Number of PCA Components')
ylabel('Accuracy (Percentage)')

%% Optimized KNN Classifier

fprintf('Creating the Optimized KNN Model...\n')

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
    Mdl = fitcknn(score, train_y, 'OptimizeHyperParameters', 'auto');
    
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
    fprintf('Optimized KNN. Num Components: %d - %.6f%% Accuracy\n', pca_count, accuracy);
    
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
title('Optimized KNN Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories')
xlabel('Number of PCA Components')
ylabel('Accuracy (Percentage)')

%% Ensemble of Learners for classification tree

fprintf('Creating the Ensemble Model...\n')

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
    fprintf('Ensemble. Num Components: %d - %.6f%% Accuracy\n', pca_count, accuracy);
    
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
title('Ensemble Classifier - Accuracy vs. Number of Components (PCA) - 20 Categories')
xlabel('Number of PCA Components')
ylabel('Accuracy (Percentage)')


%% Cleanup

fprintf('Completed the Supervised Model Test.\n')



