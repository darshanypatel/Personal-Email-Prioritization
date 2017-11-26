%% Setup

clc; clear; close all

%% Analyzer

TOP = '/Users/alperender/Desktop/CSC522Project/raw_data_files';
cd(TOP)

% Initializing DATA
DATA = cell(100,6);
counter = 1;

% Invalid file/folder name
invalid = {'.','..','.DS_Store'};
        
% Obtaining all the files for all users
files = dir;

% Iterating through all files for all users
for i = 1:length(files)

    % Obtaining the file name
    fileName = files(i).name;

    %{
        --- EXECUTE ACTIONS ON FOLDERS HERE ---
    %}

    % Obtaining all the sent folder names
    if ~sum(strcmpi(fileName,invalid))

        % Obtaining Information from each file
        FID = fopen(fileName,'r');
        
        rawID = fgetl(FID);
        rawDate = fgetl(FID);
        rawFrom = fgetl(FID);
        rawTo = fgetl(FID);
        rawSubject = fgetl(FID);
        
        ID = strip(rawID(12:end));
        Date = strip(rawDate(6:end));
        From = strip(rawFrom(6:end));
        To = rawTo(4:end);
        Subject = strip(rawSubject(9:end));
        
        tok = find(To==',');
        
        if ~isempty(tok)
            To = strip(split(To,','));
        else
            To = strip({To});
        end
        
        message = fileread(fileName);
        startLoc = regexp(message,'X-FileName: ');
        endLoc = find(message == char(10));
        loc = endLoc(endLoc > startLoc);
        message(1:loc(1)) = [];
                        
        fclose(FID);
        
        DATA{counter,1} = ID;
        DATA{counter,2} = Date;
        DATA{counter,3} = From;
        DATA{counter,4} = To;
        DATA{counter,5} = Subject;
        DATA{counter,6} = message;
        
        counter = counter + 1;
        
    end

end

%%

clear files
      
