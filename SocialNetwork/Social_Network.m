%% Reading in the CSV File

% Opening the CSV File
FID = fopen('/Users/alperender/Desktop/ALDA-Project/Unsupervised Final/Unsupervised Output.csv','r');

% Initializing values
data = {};
i = 1;

% Looping through the file
while ~feof(FID)
    
    % Obtaining each line
    entry = fgetl(FID);
    
    % Splitting the lines based on commas
    data(i,:) = split(entry,',');
    
    % Updating counter
    i = i + 1;
    
    disp(i)
    
end

% Closing File
fclose(FID);


%% Setting up Data

from = data(:,2);
to = data(:,3);

%% Graphs

% Setting up weights
% TO_WEIGHT  = 10;
% CC_WEIGHT  = 7;
% BCC_WEIGHT = 5;
%
% Anonymous function to get weight of each email
% GetWeight = @(x) strcmpi(x,'TO') * TO_WEIGHT + strcmpi(x,'CC') * CC_WEIGHT + strcmpi(x,'BCC') * BCC_WEIGHT;

% Setting up graph
G = digraph();

% Iterating through number of sent values
for i = 1:length(from)
    
    % Initializing to values and splitting based on delimiter :
    to_vals = {};
    to_emails = split(to{i}, ':');
    
    % Iterating through TO values
    for j = 1:length(to_emails)
        
        % If the value is not empty from splitting
        if ~isempty(to_emails{j})
            
            % Obtain the TO email
            to_vals{j,1} = strip(to_emails{j});
            
            % fprintf('%s - %s \n', from{i}, to_vals{j,1})
        end
    end
    
    % If this is the first time running through the loop
    if i == 1
        
        % Add the first node to the graph
        G = addnode(G, from(i));
        
        % Obtain the number of TO emails and loop through each one
        [r,c] = size(to_vals);
        for j = 1:r
            
            % Add the TO email as a node
            G = addnode(G, to_vals{j,1});
            
            % Add an edge from the FROM email to the TO email with the
            % weight specified above
            G = addedge(G, from{i}, to_vals{j,1}, 1);
            
        end
        
        % Continue to the next iteration
        continue
    end
    
    % All other iterations pass through here
    % If the FROM node is not in the graph, add it
    if ~findnode(G,  from{i})
        G = addnode(G, from{i});
    end
    
    % Obtain the size of the TO emails and loop through them
    [r,c] = size(to_vals);
    for j = 1:r
        
        % If the TO email is not in the graph, add it
        if ~findnode(G, to_vals{j,1})
            G = addnode(G, to_vals{j,1});
        end
        
        % If the edge of the FROM node to the TO node is NOT in the graph
        % (OR VICE VERSA), add it with the weight specified above
        if ~findedge(G, from{i}, to_vals{j,1})
            
            G = addedge(G, from{i}, to_vals{j,1}, 1);
            
        else
            
            % If the edge of the FROM node to the TO node IS in the graph,
            % (OR VICE VERSA), then obtain the weight of the edge, and then
            % add the new weight to the edge and store it back in the graph
            G.Edges.Weight(findedge(G, from{i}, to_vals{j,1})) = G.Edges.Weight(findedge(G, from{i}, to_vals{j,1})) + 1;
            
        end
    end
end


%% Obtaining adjancy graph

lookup_table = {};

% Creating the nodes from the graph
for i = 1:height(G.Nodes)
    val = table2cell(G.Nodes(i,1));
    lookup_table{i,1} = val{1};
    lookup_table{i,2} = i;
end

% ROWS - Sender
% COLUMNS - Receiver
% Creating a zeros matrix for the nodes
adjancy_mat = zeros(height(G.Nodes));

% Obtaining all the edges
edges = table2cell(G.Edges);

% Iterating through all the edges
for i = 1:length(edges)
    
    % The finding the names in the edges in the lookup table
    rowVal = find(strcmpi(edges{i,1}{1}, lookup_table(:,1)));
    colVal = find(strcmpi(edges{i,1}{2}, lookup_table(:,1)));
    
    % Putting the weighted values into the matrix
    adjancy_mat(rowVal,colVal) = edges{i,2};
    
end

% Turning the matrix into a sparse matrix
s_adjency_mat = sparse(adjancy_mat)


%% Obtaining a Second Graph

mat_2 = adjancy_mat;
[r,c] = size(mat_2);

for i = 1:r
    for j = 1:c
        
        if j == i
            break
        end
        
        mat_2(i,j) = mat_2(i,j) + mat_2(j,i);
        mat_2(j,i) = mat_2(i,j);
        
    end
end

G2 = graph(mat_2, G.Nodes);

%% Obtaining number of neighbors

% Initializing variables
neigh = {};

% Looping through the number of nodes in the graph
for i = 1:height(G2.Nodes)
    
    % Obatining the name of the node and storing it in the cell
    val = table2cell(G2.Nodes(i,1));
    neigh{i,1} = val{1};
    
    % Obtaining the number of neighbors of the node and storing it in the
    % cell
    neigh{i,2} = length(neighbors(G2, table2array(G2.Nodes(i,1))));
    
end

%% Setting up categories matrix

% user_categories is the number of total emails per user depending on the
% number of categories
% ROWS - the users
% COLUMNS - the categories from the unsupervised model
% CELLS - number of emails for that user in that category

% search_lookup takes a search(in) and finds the index at which the name is
% found in the lookup table
search_lookup = @(in) find(strcmpi(in, lookup_table(:,1)));

% Setting up matrix where the rows are the number of users and the columns
% are the max number of groups
user_categories = zeros(length(neigh), max(str2double(data(:,end))));

% Iterating through the data
for i = 1:length(data)
    
    % Obtaining the category of the email - also is the column of the
    % matrix
    ind_col = str2double(data{i,end});
    
    % Obtaining the sender
    sender = data{i,2};
    
    % Looking up the sender in the lookup table
    ind_row = search_lookup(sender);
    
    % Updating the user_categories matrix with the index of the sender
    user_categories(ind_row, ind_col) = user_categories(ind_row, ind_col) + 1;
    
    % Obtaining the receiver
    receiver = data{i,3};
    
    % Splitting the receivers based on the delimiter
    receivers = split(receiver, ':');
    
    for j = 1:length(receivers)
        
        % Looking up the receiver in the lookup table
        ind_row = search_lookup(receivers{j});
        
        % Updating the user_categories matrix with the index of the receiver
        user_categories(ind_row, ind_col) = user_categories(ind_row, ind_col) + 1;
        
    end
end

%% Set up emails matrix

% user_emails is the number of total emails per user
% ROWS - each email user
% COLUMNS - 1 column that holds the number of emails for that user

% Setting up matrix
user_emails = zeros(length(neigh), 1);

% Iterating through the data
for i = 1:length(data)
    
    % Obtaining the sender and the reciever
    sender = data{i,2};
    
    % Looking up the sender in the lookup table
    ind_row = search_lookup(sender);
    
    % Updating the user_emails matrix with the index of the sender
    user_emails(ind_row, 1) = user_emails(ind_row, 1) + 1;
    
    % Obtaining the receiver
    receiver = data{i,3};
    
    % Splitting the receivers based on the delimiter
    receivers = split(receiver, ':');
    
    for j = 1:length(receivers)
        
        % Looking up the receiver in the lookup table
        ind_row = search_lookup(strip(receivers{j}));
        
        % Updating the user_categories matrix with the index of the receiver
        user_emails(ind_row, 1) = user_emails(ind_row, 1) + 1;
        
    end
    
end


%% Calculating the Importance Rating

% WEIGHTS
w1 = 0.6;
w2 = 0.3;
w3 = 0.1;

% Setting up the importance calculation of the email
importance = {};

% Finding the maximum edges for all nodes
max_edges_all_users = max(cell2mat(neigh(:,2)));

% Iterathing through all the messages
for i = 1:length(data)
    
    % Finding the category for each email
    category = str2double(data{i,end});
    
    % Obtaining the sender
    sender = data{i,2};
    
    % Looking up the sender
    ind_row = search_lookup(sender);
    
    % Obtaining the reciever
    receivers = data{i,3};
    
    % Splitting receivers based on delimiters
    receivers_cell = split(receivers, ':');
    
    % Iterating through all the receivers
    for j = 1:length(receivers_cell)
        
        % Obtaining the receiver
        receiver = strip(receivers_cell{j});
        
        % Don't consider empty cells
        if isempty(receiver)
            continue
        end
        
        % Looking up the receiver
        ind_col = search_lookup(receiver);
        
        % --- Formula 1 ---
        
        % Obtain the total number of emails for that category
        top = user_categories(ind_col, category);
        
        % Obtain the total number of emails for that user
        bottom = user_emails(ind_col,1);
        
        % Implementing formula 2
        val_1 = top / bottom;
        
        % --- Formula 2 ---
        
        % Obtaining the messages between users by looking at the adjancy matrix
        messages_bt_users = adjancy_mat(ind_row, ind_col) + adjancy_mat(ind_col, ind_row);
        
        % Implementing formula 2
        val_2 = ( messages_bt_users - abs( adjancy_mat(ind_row, ind_col) - adjancy_mat(ind_col, ind_row) ) ) / messages_bt_users;
        
        % --- Formula 3 ---
        
        % Obtaining the number of neighbor nodes the sender has
        sender_num_nodes = length(neighbors(G2, sender));
        
        % Obtaining the number of
        val_3 = sender_num_nodes / max_edges_all_users;
        
        % Calculating the importance
        importance_calc = w1 * val_1 + w2 * val_2 + w3 * val_3;
        
        importance{i,j} = importance_calc;
        
    end
    
end
