%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% @AutoGenerated
%
% Filename: main.m
% Author: Alper Ender
% Date: November 2017
% Description:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\

%% Setup

% Clear work station
clc; clear; fclose('all')

% Turn off warnings
warning('off')

% Change directory to current file folder
cd(fileparts(which(mfilename)))

if isunix
    slash = '/';
else
    slash = '\';
end

% Obtain the top folder path
folder_system.top = [pwd slash];

% Iterate through all the folders
folders = dir;
for i = 1:length(folders)
    
    % Ensure that the folder is valid
    if isdir(folders(i).name) & ~strcmpi(folders(i).name(1), '.')
        
        % Change into the folder, store the path and return
        cd(folders(i).name)
        folder_system.(folders(i).name) = [pwd slash];
        cd(folder_system.top)
        
    end
end

% Adding directory paths to MATLAB search path
paths = struct2cell(folder_system);
all_paths = [];

for i = 1:length(paths)
    if isunix
        all_paths = [all_paths ':' paths{i}];
    else
        all_paths = [all_paths ';' paths{i}];
    end
end
all_paths(1) = [];
addpath(all_paths)


%%

% Step2_Divider

% cd('Unsupervised Final')
%
% Unsupervised_Models_Final
%
% cd('../')
%
% cd('Supervised Test')
%
% Split_Data
%
% Supervised_Model_Test

%% Cleanup

% rmpath(all_paths)