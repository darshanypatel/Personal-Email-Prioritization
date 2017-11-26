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
            
            % Obtaining all the discussion folder names
            if contains(lower(folderName),'discussion')
                
                % Deleting the discussion folders
                fprintf('Deleting: %s\n', currentF)
                rmdir(folderName,'s')
                
            end
            
        end
        
        cd('../')
    end
end

