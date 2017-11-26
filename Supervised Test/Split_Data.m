% Author: Alper Ender

clc; clear; fclose('all')

% Opening the file to read
FID = fopen('/Users/alperender/Desktop/ALDA-Project/Unsupervised Final/Unsupervised Output.csv','r');

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

%% Obtaining the Training Set and the Test Set

% Percent of the values that are trained
PERCENT_TRAIN = 80;

% Obtain the number of emails
counter = length(all_emails);

% Create a random permuation from 1 to the number of emails counted
picked_vals = randperm(counter);

% Obtain the number of values to obtain for the training dataset 
train_num = floor((PERCENT_TRAIN/100) * counter);

% Obtain the training and testing email index numbers
train_vals = picked_vals(1:train_num);
test_vals = picked_vals(train_num+1:end);

% Obtain the training and testing emails
train_emails = all_emails(train_vals,:);
test_emails = all_emails(test_vals,:);


%% Export and Write to file

% Write the training data to file
fid = fopen('Train.csv','w');

% Obtaining the size of the training set
[r,c] = size(train_emails);

% Going through each email
for i = 1:r
    
    % Going through each field
    for j = 1:c
        
        % Print the value to file
        fprintf(fid, '%s', train_emails{i,j});
        
        % Do NOT print a comma for the last value on a line
        if j ~= c
            fprintf(fid, ',');
        end
    end
    
    % Do NOT print a linebreak for the last email
    if i ~= length(train_emails)
        fprintf(fid, '\n');
    end
end

% Close the file
fclose(fid);


% Write the testing set to the file
fid = fopen('Testing.csv','w');

% Obtaining the size of the training set
[r,c] = size(test_emails);

% Going through each email
for i = 1:r
    
    % Going through each field
    for j = 1:c
        
        % Print the value to file
        fprintf(fid, '%s', test_emails{i,j});
        
        % Do NOT print a comma for the last value on a line
        if j ~= c
            fprintf(fid, ',');
        end
    end
    
    % Do NOT print a linebreak for the last email
    if i ~= length(test_emails)
        fprintf(fid, '\n');
    end
end

% Close the file
fclose(fid);



