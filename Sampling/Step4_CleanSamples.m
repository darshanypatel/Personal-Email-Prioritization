clc; clear; fclose('all')

% Defining files to parse
files = dir('*.csv');

% Iterating through files
for i = 1:length(files)
    
    % Setting the filename
    filename = files(i).name
    
    % Opening the filename to read from
    FID = fopen(filename, 'r');
    
    % Opening file to write to
    FID_write = fopen(['CleanBody_' filename], 'w');
    
    % Iterate through the CSV file
    while ~feof(FID)
        
        % Get the line and tokenize based on delimiter
        line = fgetl(FID);
        tok = textscan(line,'%s','Delimiter',',');
        
        % Parse the body from the tokens
        body = tok{1}{end};
        
        % Regexp the body for the .nsf or .pst location
        [token, start, e] = regexp(lower(body), '(\.(?:nsf|pst))','tokens','once');
        
        % If the body is empty, ignore it
        if isempty(body(e-40:e+1))
            continue
        end
        
        % Storing the tokens
        a = tok{1};
        
        % Strip the first part of the body
        fprintf(FID_write, '%s,%s,%s,%s,%s,%s\n', a{2}, a{3}, a{4}, a{5}, a{6}, strip(body(e+1:end)));
        
    end
    
    % Close files
    fclose(FID);
    fclose(FID_write);
    
end