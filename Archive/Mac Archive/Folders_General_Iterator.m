clc; clear; close all

%% Analyzer

TOP = '/Users/alperender/Desktop/CSC522Project/raw_data_folders';
cd(TOP)

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
            currentF = [currentU '/' folderName];
            
            %{
                --- EXECUTE ACTIONS ON FOLDERS HERE ---
            %}
            
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
                    currentFI = [currentF '/' fileName];
                    
                    % If the filename is valid
                    if ~sum(strcmpi(fileName,invalid)) & ~isdir(fileName)
                        
                        %{
                            --- EXECUTE ACTIONS ON FILES HERE ---
                        %}
                        
                        disp(currentFI)
                        
                        % words = fileread(fileName);
                        
                        % if contains(lower(words), 'no.address@enron.com')
                        %     disp(words)
                        % end
                        
                    end
                    
                end
                
                cd('../')
            end
     
        end
        
        cd('../')
    end
end


