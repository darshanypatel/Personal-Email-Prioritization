% Author: Alper Ender
% Date: November 2017
% Description: Takes in the large enron csv email file and counts the
% number of emails and number of unique users in the corpus

% TOTAL EMAILS: 490854
% TOTAL EMAIL ACCOUNTS: 149

clc;

% Opening the file to read
FID = fopen('/Users/alperender/Downloads/Enron_Emails_Uncleaned.csv','r');

% Initalizing variables
counter = 1;
all_names = {};

% Running through the file
while ~feof(FID)
    
    % Read the line
    line = fgetl(FID);
    
    % Tokenize the line from the delimiter
    tokens = textscan(line,'%s','Delimiter',',');
    
    % Obtain the name from the line
    tokenized_name = tokens{1}{6};
    
    % The name is delimited by an underscore
    loc = find(tokenized_name == '_');
    
    % Don't let the underscore be the first character
    ind = 1;
    if loc(1) == 1
        ind = 2;
    end
    
    % Obtain the name from the token
    name = tokenized_name(1:loc(ind)-1);
    
    % Store the name
    all_names{counter,1} = name;
    
    % Display counter
    if mod(counter,10000) == 0
        disp(counter)
    end
    
    % Increment counter
    counter = counter + 1;
    
end

%% Finding the unique names

% Find all the unique names and find the total count of names
unique_names = unique(all_names);
unique_count = length(unique_names);