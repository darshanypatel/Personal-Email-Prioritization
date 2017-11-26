clc;

%% Setup

% TOTAL EMAILS: 490854
% TOTAL EMAIL ACCOUNTS: 149

% Folder location
FOLDER_LOCATION = '/Users/alperender/Desktop/ALDA-Project/Individual Files';
TOTAL_SAMPLE = 400000;

FILE_LOCATION = '/Users/alperender/Desktop/ALDA-Project/';

write_FID = fopen([FILE_LOCATION 'tmp.csv'],'w');

%% Calculations

total_emails = 490854;
total_email_accounts = 149;

samples_per_student = floor(TOTAL_SAMPLE / total_email_accounts)

all_emails = cell(samples_per_student*total_email_accounts,1);
ae_counter = 1;

%% Obtaining samples

email_samples = {};

% Change to the folder location
cd(FOLDER_LOCATION)

% Getting all the students in the folder
students = dir('*.csv');

% Iterating through all the csv files
for i = 1:length(students)
    
    % The name of the student
    name = students(i).name
    
    % Opening the file
    FID = fopen(name);
    
    % Resetting the variables
    student_emails = {};
    count = 1;
    
    % Obtaining all the emails from the file
    while ~feof(FID)
        
        line = fgetl(FID);
        tok = split(line,',');
        
        % Looking for only inbox items with subject without .com inside of
        % the subject and sion: 1.0 (a bug)
        if contains(lower(line), '_inbox') & ~contains(tok{5},'.com') & ~contains(tok{5}, 'sion: 1.0') & ~contains(tok{4}, 'ject:')
            student_emails{count, 1} = line;
            count = count + 1;
        end
        
    end
    
    % Obtain the size of all the emails
    [r,c] = size(student_emails);
    
    % Create a random set of numbers
    r_values = randperm(r);
    
    % Take the lesser between r and and samples per student
    if samples_per_student > r
        ind = r;
    else
        ind = samples_per_student;
    end
    
    % Take the first n numbers
    r_values_person = r_values(1:ind);
    
    % Take the first n eamils
    r_emails = student_emails(r_values_person);
    
    % Append the emails to the file
    for j = 1:length(r_emails)
        if isempty(r_emails{j})
            continue
        end
        
        if i == length(students) & j == length(r_emails)
            fprintf(write_FID, '%s', r_emails{j});
        else
            fprintf(write_FID, '%s\n', r_emails{j});
        end
    end
    
    fclose(FID);
    
end

fclose('all')

%% Rename the file appropriately

FID = fopen([FILE_LOCATION 'tmp.csv'], 'r');

counter = 0;

while ~feof(FID)
    fgetl(FID);
    counter = counter + 1;
end

fclose(FID);

movefile([FILE_LOCATION 'tmp.csv'], [FILE_LOCATION 'Samples_' int2str(counter) '.csv'])