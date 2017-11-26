%% Setup

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
    if contains(lower(fileName),'sent') & ~strcmpi(fileName, 'presentations')

        % Obtaining Information from each file
        FID = fopen(fileName,'r');
        
        rawID = fgetl(FID);
        rawDate = fgetl(FID);
        rawFrom = fgetl(FID);
        rawTo = fgetl(FID);
        rawSubject = fgetl(FID);
        
        ID = rawID(12:end);
        Date = rawDate(6:end);
        From = rawFrom(6:end);
        To = rawTo(4:end);
        Subject = rawSubject(9:end);
        
        ToAt = find(To == '@');
        if length(ToAt > 1)
        
            disp(ID)
            disp(Date)
            disp(From)
            disp(To)
            disp(Subject)
            fprintf('\n')
            
        end
        
        fclose(FID);
        
    end

end
       
%% Cleanup

clear

