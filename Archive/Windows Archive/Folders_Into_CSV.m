clc; clear; close all

%% Analyzer

TOP = 'C:\Users\Alper Ender\Desktop\CSC522Project\raw_data_folders';
cd(TOP)

FIDCSV = fopen('DATA.csv','w');

% Obtaining all users
users = dir;

% Invalid file/folder name
invalid = {'.','..','.DS_Store'};


% Iterating through all users
for i = 1:length(users)
    
    % Obtaining the user name
    userName = users(i).name;
    currentU = userName;
    
    %{
        --- EXECUTE ACTIONS ON USERS HERE ---
    %}
    
    % Ensuring that the filenames don't have these traits
    if ~sum(strcmpi(userName,invalid)) & isdir(userName)
        
        % Going into the user folder
        cd(userName)
        
        % Obtaining all the folders for the user
        folders = dir;
        
        % Iterating through all folders for the user
        for j = 1:length(folders)
            
            % Obtaining the folder name
            folderName = folders(j).name;
            currentF = [currentU '_' folderName];
            
            %{
                --- EXECUTE ACTIONS ON FOLDERS HERE ---
            %}
            
            if ~sum(strcmpi(folderName,invalid)) & ~isdir(folderName)
                
                disp(currentF);
                
                % SPECIAL FILES
                FID = fopen(folderName,'r');

                rawID      = fgetl(FID);
                rawDate    = fgetl(FID);
                rawFrom    = fgetl(FID);
                rawTo      = fgetl(FID);
                rawSubject = fgetl(FID);

                Message = fileread(folderName);

                ID      = strip(rawID(12:end));
                Date    = strip(rawDate(6:end));
                From    = strip(rawFrom(6:end));
                To      = strip(rawTo(4:end));
                Subject = strip(rawSubject(9:end));
                Location= currentF;

                Date = strrep(Date, ',', ' ');

                From = strrep(From, ',', ':');
                
                To = strrep(To, ',', ':');

                Message = strrep(Message, char(10), ' ');
                Message = strrep(Message, char(13), ' ');
                Message = strrep(Message, ',', ' ');
                
                Subject = strrep(Subject, ',', ' ');

                fprintf(FIDCSV, '%s,%s,%s,%s,%s,%s,%s\n', ID, Date, From, To, Subject, Location, Message);
                
                fclose(FID);
                
            end
            
            % Ensuring that the filenames don't have these traits
            if ~sum(strcmpi(folderName,invalid)) & isdir(folderName)
                
                % Changing directory to the file
                cd(folderName)
                
                % Obtaining all the files in the directory
                files = dir;
                
                % Iterating through all the files
                for k = 1:length(files)
                    
                    % Obtaining the filename
                    fileName = files(k).name;
                    currentFI = [currentF '_' fileName];
                    
                    % If the filename is valid
                    if ~sum(strcmpi(fileName,invalid)) & ~isdir(fileName)
                        
                        %{
                            --- EXECUTE ACTIONS ON FILES HERE ---
                        %}
                        
                        disp(currentFI);
                        
                        FID = fopen(fileName,'r');
                        
                        rawID      = fgetl(FID);
                        rawDate    = fgetl(FID);
                        rawFrom    = fgetl(FID);
                        rawTo      = fgetl(FID);
                        rawSubject = fgetl(FID);
                        
                        Message = fileread(fileName);
                        
                        ID      = strip(rawID(12:end));
                        Date    = strip(rawDate(6:end));
                        From    = strip(rawFrom(6:end));
                        To      = strip(rawTo(4:end));
                        Subject = strip(rawSubject(9:end));
                        Location= currentFI;
                        
                        Date = strrep(Date, ',', ' ');
                        
                        From = strrep(From, ',', ':');
                        
                        To = strrep(To, ',', ':');
                        
                        Message = strrep(Message, char(10), ' ');
                        Message = strrep(Message, char(13), ' ');
                        Message = strrep(Message, ',', ' ');
                        
                        Subject = strrep(Subject, ',', ' ');
                        
                        fprintf(FIDCSV, '%s,%s,%s,%s,%s,%s,%s\n', ID, Date, From, To, Subject, Location, Message);
                        
                        fclose(FID);
                        
                    end
                    
                end
                
                cd('../')
            end
     
        end
        
        cd('../')
    end
end

fclose('all')


