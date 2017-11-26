clc; clear; fclose('all')

% Opening the file to read
FID = fopen('C:\Users\Alper Ender\Downloads\Version 2\Unsupervised Final\Unsupervised Output.csv','r');

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

%%

% Reading in the dictionary words and tokenizing the words
dictionary = fileread('C:\Users\Alper Ender\Downloads\Version 2\Unsupervised Final\words.txt');
dict = tokenizedDocument(dictionary);

% Creating a bag of words based off the dictionary values
t_bag = bagOfWords(dict)

%%

docs = tokenizedDocument;

[r, c] = size(all_emails);

for i = 1:r
    docs(i,1) = tokenizedDocument(all_emails{i,end-1});
end

docs_s = encode(t_bag, docs);
all_docs = full(docs_s);

%%

num_folds = 10;
pca_count = 30;
ind = crossvalind('kfold', r, num_folds);

comp_cnb = {};
comp_bt = {};
comp_knn = {};

sse_cnb = 0;
sse_bt = 0;
sse_knn = 0;

for i = 1:num_folds
    
    disp(i)
    
    % Obtaining the training values for this fold
    train_x = all_docs(ind ~= i, :);
    train_y = all_emails(ind ~= i, end);
    
    % Obtaining the testing values for this fold
    test_x = all_docs(ind == i, :);
    test_y = all_emails(ind == i, end);
    
    % Reducing dimensionality through PCA
    [coeff,score,latent,tsquared,explained,mu] = pca(train_x, 'NumComponents', pca_count);
    mean_train_x = mean(train_x);
    
    % --- NAIVE BAYES ---
    
    % Creating the naive bayes model model
    Mdl = fitcnb(score,train_y);
    
    % Modfiying the testing data using the PCA values
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the values using the testing dataset
    predicted_vals = Mdl.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    
    % Calculating SSE
    sse_cnb = [sse_cnb sum(comparison == 0) .^ 2];
    
    % Storing the values
    comp_cnb{i} = comparison';
    
    % Printing values
    fprintf('Round %d, NB,  Training Accuracy = %%%.6f, Testing Accuracy = %%%.6f\n', i, (1 - Mdl.resubLoss('LossFun','ClassifErr')) * 100,  sum(comparison) / length(comparison) * 100);
    
    
    % --- Classification Tree ---
    
    % Creating a binary tree classifier model
    Mdl = fitctree(score, train_y);
    
    % Reshaping the testing dataset based upon the pca coefficients
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the modified values
    predicted_vals = Mdl.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    
    % Calculating SSE
    sse_bt = [sse_bt sum(comparison == 0) .^ 2];
    
    % Storing the values
    comp_bt{i} = comparison';
    
    % Printing values
    fprintf('Round %d, CT,  Training Accuracy = %%%.6f, Testing Accuracy = %%%.6f\n', i, (1 - Mdl.resubLoss('LossFun','ClassifErr')) * 100,  sum(comparison) / length(comparison) * 100);
    
    
    % --- KNN ---
    
    % Creating a KNN classifier model
    Mdl = fitcknn(score, train_y);
    
    % Reshaping the testing dataset based upon the pca coefficients
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the modified values
    predicted_vals = Mdl.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    
    % Calculating SSE
    sse_knn = [sse_knn sum(comparison == 0) .^ 2];
    
    % Storing the values
    comp_knn{i} = comparison';
    
    % Printing values
    fprintf('Round %d, KNN, Training Accuracy = %%%.6f, Testing Accuracy = %%%.6f\n', i, (1 - Mdl.resubLoss('LossFun','ClassifErr')) * 100,  sum(comparison) / length(comparison) * 100);
    
    
end

%% Optimized Supervised Values

diary('Optimized Comparison.txt');

comp_opt_nb = {};
comp_opt_ct = {};
comp_opt_knn = {};
comp_opt_ens = {};

sse_opt_nb = [];
sse_opt_ct = [];
sse_opt_knn = [];
sse_opt_ens = [];

for i = 1:num_folds
    
    % Obtaining the training values for this fold
    train_x = all_docs(ind ~= i, :);
    train_y = all_emails(ind ~= i, end);
    
    % Obtaining the testing values for this fold
    test_x = all_docs(ind == i, :);
    test_y = all_emails(ind == i, end);
    
    % Reducing dimensionality through PCA
    [coeff,score,latent,tsquared,explained,mu] = pca(train_x, 'NumComponents', pca_count);
    mean_train_x = mean(train_x);
    
    % --- NAIVE BAYES ---
    
    % Creating the naive bayes model model
    Mdl = fitcnb(score, train_y, 'OptimizeHyperParameters', 'auto');
    
    % Modfiying the testing data using the PCA values
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the values using the testing dataset
    predicted_vals = Mdl.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    
    % Calculating SSE
    sse_opt_nb = [sse_opt_nb sum(comparison == 0) .^ 2];
    
    % Storing the values
    comp_opt_nb{i} = comparison';
    
    % Printing values
    fprintf('Round %d, NB,  Training Accuracy = %%%.6f, Testing Accuracy = %%%.6f\n', i, (1 - Mdl.resubLoss('LossFun','ClassifErr')) * 100,  sum(comparison) / length(comparison) * 100);
    
    
    % --- Classification Tree ---
    
    % Creating a binary tree classifier model
    Mdl = fitctree(score, train_y, 'OptimizeHyperParameters', 'auto');
    
    % Reshaping the testing dataset based upon the pca coefficients
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the modified values
    predicted_vals = Mdl.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    
    % Calculating SSE
    sse_opt_ct = [sse_opt_ct sum(comparison == 0) .^ 2];
    
    % Storing the values
    comp_opt_ct{i} = comparison';
    
    % Printing values
    fprintf('Round %d, CT,  Training Accuracy = %%%.6f, Testing Accuracy = %%%.6f\n', i, (1 - Mdl.resubLoss('LossFun','ClassifErr')) * 100,  sum(comparison) / length(comparison) * 100);
    
    
    % --- KNN ---
    
    % Creating a KNN classifier model
    Mdl = fitcknn(score, train_y, 'OptimizeHyperParameters', 'auto');
    
    % Reshaping the testing dataset based upon the pca coefficients
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the modified values
    predicted_vals = Mdl.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    
    % Calculating SSE
    sse_opt_knn = [sse_opt_knn sum(comparison == 0) .^ 2];
    
    % Storing the values
    comp_opt_knn{i} = comparison';
    
    % Printing values
    fprintf('Round %d, KNN, Training Accuracy = %%%.6f, Testing Accuracy = %%%.6f\n', i, (1 - Mdl.resubLoss('LossFun','ClassifErr')) * 100,  sum(comparison) / length(comparison) * 100);
    
    % --- Ensemble ---
    
    % Creating an ensemble model with multiple parameters
    templ = templateTree('Surrogate','all');
    ens = fitcensemble(score, train_y,'OptimizeHyperparameters','auto');
    
    % Reshaping the testing dataset based upon the pca coefficients
    modified_test_x = (test_x - mean_train_x) * coeff;
    
    % Predicting the modified values
    predicted_vals = ens.predict(modified_test_x);
    
    % Calculating the number correct and incorrect classifications
    comparison = str2double(test_y) == str2double(predicted_vals);
    
    % Calculating SSE
    sse_opt_ens = [sse_opt_ens sum(comparison == 0) .^ 2];
    
    % Storing the values
    comp_opt_ens{i} = comparison';
    
    % Printing values
    fprintf('Round %d, ENS, Training Accuracy = %%%.6f, Testing Accuracy = %%%.6f\n', i, (1 - Mdl.resubLoss('LossFun','ClassifErr')) * 100,  sum(comparison) / length(comparison) * 100);
    
end

diary(off)

%% Plotting results

figure();

% Plotting the Squared Error values
x_data = 1:num_folds;
scatter(x_data, sse_opt_nb, '*b')
hold on
scatter(x_data, sse_opt_ct, '*r')
scatter(x_data, sse_opt_knn, '*g')
scatter(x_data, sse_opt_ens, '*k')

% Plotting the MSE values
x_vals =  [min(x_data)-0.5, max(x_data)+0.5];

% Plotting the MSE for Naive Bayes
mse_sse = mean(sse_opt_nb);
plot(x_vals, [mse_sse mse_sse], 'b')
text(x_vals(end), mse_sse, sprintf('MSE = %.f', mse_sse), 'color','b')

% Plotting the MSE for Binary Classifier
mse_sse = mean(sse_opt_ct);
plot(x_vals, [mse_sse mse_sse], 'r')
text(x_vals(end), mse_sse, sprintf('MSE = %.f', mse_sse), 'color','r')

% Plotting the MSE for KNN
mse_sse = mean(sse_opt_knn);
plot(x_vals, [mse_sse mse_sse], 'g')
text(x_vals(end), mse_sse, sprintf('MSE = %.f', mse_sse), 'color','g')

% Plotting the MSE for Ensemble
mse_sse = mean(sse_opt_ens);
plot(x_vals, [mse_sse mse_sse], 'k')
text(x_vals(end), mse_sse, sprintf('MSE = %.f', mse_sse), 'color','k')

% Formatting plot
grid on
legend('Naive Bayes', 'Binary Tree', 'KNN', 'Ensemble')
xlabel('K-Fold')
ylabel('Squared Error')
xlim( [ min(x_data)-0.5, max(x_data)+0.5 ] )
title('10 Fold Cross Validation - Mean Squared Error')
