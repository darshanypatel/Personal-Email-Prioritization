clc; clear; close all

%% Analyzer

TOP = '/Users/alperender/Desktop/CSC522Project/raw_data_files';
cd(TOP)

% Invalid file/folder name
invalid = {'.','..','.DS_Store'};
        
% Obtaining all the files for all users
files = dir;

% Iterating through all files for all users
for j = 1:length(files)

    % Obtaining the file name
    fileName = files(j).name;

    %{
        --- EXECUTE ACTIONS ON FOLDERS HERE ---
    %}

    % Obtaining all the sent folder names
    if contains(lower(fileName),'discussion')

        % Deleting the sent folders
        fprintf('Deleting: %s\n', fileName)
        % delete(fileName)

    end

end

%% 

clear;
        

