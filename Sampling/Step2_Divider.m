% Author: Alper Ender
% Date: November 2017
% Description: Takes in the large enron csv email file and split it based
% on the name of the folder the email is found

clc;

% Open the entire email file
FID = fopen('/Users/alperender/Desktop/ALDA-Project/Enron_Emails_Uncleaned.csv','r');

% Set up initial variables
current_name = '';
current_file = [];

% Iterate through the entire file
while ~feof(FID)
    
    % Obtain the line
    line = fgetl(FID);
    
    % Tokenize the line based on the delimiter
    tokens = split(line,',');
    
    % Obtain the folder/file name
    token_name = tokens{6};
    
    % Find the first underscore
    loc = find(token_name == '_');
    ind = 1;

    % If the underscore is at the beginning, find the second underscore
    if loc(1) == 1
        ind = 2;
    end
    
    % Obtain the name
    name = token_name(1:loc(ind)-1);
    
    % Go until a name change
    if ~strcmpi(name, current_name)
        
        % Close the old file
        try
            fclose(current_file);
        catch
        end
        
        % Change the filename
        current_name = name;
        
        % Open the new file FID
        current_file = fopen([current_name '.csv'],'w');
        
        % Display the name
        disp(name)
    end
    
    % Print each line to the correct file
    fprintf(1, '%s\n',line);
    fprintf(current_file, '%s\n',line);
    
end

fclose('all')