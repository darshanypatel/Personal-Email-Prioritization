%% Notes

%{

Important Links (SSE):
    http://www.cs.uky.edu/~jzhang/CS689/PPDM-Chapter3.pdf
    https://www.cse.buffalo.edu/~jing/cse601/fa12/materials/clustering_basics.pdf
    https://hlab.stanford.edu/brian/error_sum_of_squares.html

%}

%% Clearing Everything

clc; clear; close all;

%% Setup

% Location of the CSV File or Folder System
FOLDER_LOCATION = '/Users/alperender/Dropbox/CSC 522/Project';  % (Mac)
% FOLDER_LOCATION = 'C:\Users\Alper Ender\Desktop\inbox';       % (Windows)

% Use the Folder System (true) or Read in a CSV File (false)
USE_FOLDER_SYSTEM = false;          

% The Filename to read in
FILENAME = 'test.csv';   

%% Read in the values

% PreProcess referenced and used from:
% https://www.mathworks.com/help/textanalytics/
%         examples/prepare-text-data-for-analysis.html

cd(FOLDER_LOCATION)

if USE_FOLDER_SYSTEM
    
    % Obtain the number of files
    files = dir;
    
    % Initializing docs and count
    docs = tokenizedDocument;
    count = 1;
    
    % Iterating through number of files
    parfor i = 1:length(files)
        
        % Obtaining the name of the file
        name = files(i).name;
        
        % If the name is not illegal
        if sum(strcmpi(name,{'.','..','.DS_STORE'}))==0
            
            % Preprocess the values
            docs(count,1) = PreProcess(fileread(name));
            
            % Increment the counter
            count = count+1;
            disp(name)
        end
        
    end
    
else
    
    FID = fopen(FILENAME,'r');
    
    % Initializing docs and count
    docs = tokenizedDocument;
    count = 1;
    
    % Iterating through number of files
    while ~feof(FID)
        
        % Reading the line
        line = fgetl(FID);
        
        % Split based on commas
        vals = split(line, ',');
        
        % Preprocess the values
        docs(count,1) = PreProcess(vals{5});
        
        % Increment the counter
        disp(count)
        count = count+1;
                
    end
    
    % Close the file
    fclose(FID);
    
end

%% Bag of Words Model

% Creating the bag of words of model
bag = bagOfWords(docs);

% Removing infrequent words
bag = removeInfrequentWords(bag,2);

% Obtaining the Number of Words and the Number of Documents
numberOfWords = bag.NumWords;
numberOfDocuments = bag.NumDocuments;


%% TFIDF Model

% Creating the tfidf model
M = tfidf(bag);

% Looking at the model
m = full(M);
topkwords(bag,10)

% Getting the size of M
size(M)
full_M = full(M);


%% Bag of Words, LDA, K-Means

% Creating an LDA model from the bag of words
numTopics = 5
mdl = fitlda(bag, numTopics)

% Defining distance algorithms
% algorithms = {'sqeuclidean', 'cityblock'};
algorithms = {'sqeuclidean', 'cityblock', 'correlation', 'cosine'};

figure()
set(gcf,'Name', 'Bag of Words, LDA, K-Means')

for i = 1:length(algorithms)
    
    % Specifying subplot values
    subplot(2,2,i)
    % subplot(1,2,i)
    
    % Grouping the documents based on k-means 
    groups = kmeans(mdl.DocumentTopicProbabilities, numTopics,...
                    'Distance'  , algorithms{i},...
                    'MaxIter'   , 1000',...
                    'Replicates', 5);

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentTopicProbabilities,'NumComponents',2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups)');
    
    title(['K-Means ' upper(algorithms{i})])
    grid on

end

% Validation using SSE for K-means of Euclidean Distance
[r, ~] = size(bag.Counts);
SSE = zeros(1,r);

%%
for i = 1:r/10
    
    % Grouping the documents based on k-means 
    [groups,C]  = kmeans(mdl.DocumentTopicProbabilities, i,...
                                             'Distance', 'sqeuclidean');
    % Obtaining the SSE
    tot = 0;
    for j = 1:length(mdl.DocumentTopicProbabilities)
        
        % Obtaining the distace of the point to the center of the cluster
        tot = tot + pdist([mdl.DocumentTopicProbabilities(j,:) ; C(groups(j),:)]);
        
    end
    
    % Storing the SSE Value
    SSE(i) = tot;
    
end

% Plotting the SSE
figure()
set(gcf,'Name', 'Bag of Words, LDA, K-Means')
plot(1:r, SSE)

grid on
title('SSE vs. Number of Partitions')
xlabel('Number of Partitions')
ylabel('SSE')


%% Bag of Words, LSA, K-Means

numTopics = 3
mdl = fitlsa(bag, numTopics)

% Defining distance algorithms
algorithms = {'sqeuclidean', 'cityblock', 'correlation', 'cosine'};

figure()
set(gcf,'Name', 'Bag of Words, LSA, K-Means')

for i = 1:length(algorithms)
    
    % Specifying subplot values
    subplot(2,2,i)
    
    % Grouping the documents based on k-means 
    groups = kmeans(mdl.DocumentScores, numTopics,...
                    'Distance'  , algorithms{i},...
                    'MaxIter'   , 1000',...
                    'Replicates', 5);

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentScores,'NumComponents',2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups)');
    
    title(['K-Means ' upper(algorithms{i})])
    grid on

end

% Validation using SSE
[r, c] = size(bag.Counts);
SSE = zeros(1,r);

for i = 1:r
    
    % Grouping the documents based on k-means 
    [groups,C]  = kmeans(mdl.DocumentScores, i,...
                                 'Distance', 'sqeuclidean');
    % Obtaining the SSE
    tot = 0;
    for j = 1:length(mdl.DocumentScores)
        
        tot = tot + pdist([mdl.DocumentScores(j,:) ; C(groups(j),:)]);
        
    end
    
    % Storing the SSE Value
    SSE(i) = tot;
    
end

% Plotting the SSE
figure()
set(gcf,'Name', 'Bag of Words, LSA, K-Means')
plot(1:r, SSE)

grid on
title('SSE vs. Number of Partitions')
xlabel('Number of Partitions')
ylabel('SSE')

%% TFIDF, LDA, K-Means

% Creating an LDA model from the bag of words
numTopics = 10
mdl = fitlda(round(full_M), numTopics)

% Defining distance algorithms
algorithms = {'sqeuclidean', 'cityblock', 'correlation', 'cosine'};

figure()
set(gcf,'Name', 'TFIDF, LDA, K-Means')

for i = 1:length(algorithms)
    
    % Specifying subplot values
    subplot(2,2,i)
    
    % Grouping the documents based on k-means 
    groups = kmeans(mdl.DocumentTopicProbabilities, numTopics,...
                    'Distance'  , algorithms{i},...
                    'MaxIter'   , 1000',...
                    'Replicates', 5);

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentTopicProbabilities,'NumComponents',2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups)');
    
    title(['K-Means ' upper(algorithms{i})])
    grid on

end

%%
% Validation using SSE
[r, c] = size(full_M);
SSE = zeros(1,r);

for i = 1:r
    
    % Grouping the documents based on k-means 
    [groups,C]  = kmeans(mdl.DocumentTopicProbabilities, i,...
                                             'Distance', 'sqeuclidean');
    % Obtaining the SSE
    tot = 0;
    for j = 1:length(mdl.DocumentTopicProbabilities)
        
        tot = tot + pdist([mdl.DocumentTopicProbabilities(j,:) ; C(groups(j),:)]);
        
    end
    
    % Storing the SSE Value
    SSE(i) = tot;
    
end

% Plotting the SSE
figure()
set(gcf,'Name', 'TFIDF, LDA, K-Means')
plot(1:r, SSE)

grid on
title('SSE vs. Number of Partitions')
xlabel('Number of Partitions')
ylabel('SSE')

%% TFIDF, LDA, K-Medoids

numTopics = 3;
mdl = fitlda(round(full_M), numTopics)

% Defining the distance algorithms to use
algorithms = {'sqeuclidean', 'cityblock', 'correlation', 'cosine'};

% Setting up figure
figure()
set(gcf,'Name', 'TFIDF, LDA, K-Medoids')

for i = 1:length(algorithms)
    
    % Specifying subplot values
    subplot(2,2,i)
    
    % Grouping the documents based on k-medoids 
    [groups,C] = kmedoids(mdl.DocumentTopicProbabilities, numTopics,...
                                            'Distance'  , algorithms{i});

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentTopicProbabilities,'NumComponents',2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups)');
    
    % Title
    title(['K-Means ' upper(algorithms{i})])
    grid on

end

% Validation using SSE
[r, c] = size(full_M);
SSE = zeros(1,r);

for i = 1:r
    
    % Grouping the documents based on k-means 
    [groups,C]  = kmedoids(mdl.DocumentTopicProbabilities, i,...
                                             'Distance'  , 'sqeuclidean');
                       
    % Obtaining the SSE
    tot = 0;
    for j = 1:length(mdl.DocumentTopicProbabilities)
        
        tot = tot + pdist([mdl.DocumentTopicProbabilities(j,:) ; C(groups(j),:)]);
        
    end

    % Storing the SSE Value
    SSE(i) = tot;
    
end

% Plotting the SSE
figure()
set(gcf,'Name', 'TFIDF, LDA, K-Medoids')
plot(1:r, SSE)

grid on
title('SSE vs. Number of Partitions')
xlabel('Number of Partitions')
ylabel('SSE')


%% TFIDF, LSA, K-Means

numTopics = 3
mdl = fitlsa(full_M, numTopics)

% Defining the distance algorithms to use
algorithms = {'sqeuclidean', 'cityblock', 'correlation', 'cosine'};

% Setting up figure
figure()
set(gcf,'Name', 'TFIDF, LSA, K-Means')

for i = 1:length(algorithms)
    
    % Specifying subplot values
    subplot(2,2,i)
    
    % Grouping the documents based on k-means 
    groups = kmeans(mdl.DocumentScores, numTopics,...
                          'Distance'  , algorithms{i},...
                          'MaxIter'   , 1000',...
                          'Replicates', 5);

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentScores,'NumComponents',2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups)');
    
    % Title
    title(['K-Means ' upper(algorithms{i})])
    grid on

end

% Validation using SSE
[r, c] = size(full_M);
SSE = zeros(1,r);

for i = 1:r
    
    % Grouping the documents based on k-means 
    [groups,C]  = kmeans(mdl.DocumentScores, i,...
                                 'Distance', 'sqeuclidean');
    % Obtaining the SSE
    tot = 0;
    for j = 1:length(mdl.DocumentScores)
        
        tot = tot + pdist([mdl.DocumentScores(j,:) ; C(groups(j),:)]);
        
    end
    
    % Storing the SSE Value
    SSE(i) = tot;
    
end

% Plotting the SSE
figure()
set(gcf,'Name', 'TFIDF, LSA, K-Means')
plot(1:r, SSE)

grid on
title('SSE vs. Number of Partitions')
xlabel('Number of Partitions')
ylabel('SSE')


%% TFIDF, LSA, K-Medoids

numTopics = 3;
mdl = fitlsa(full_M, numTopics)

% Defining the distance algorithms to use
algorithms = {'sqeuclidean', 'cityblock', 'correlation', 'cosine'};

% Setting up figure
figure()
set(gcf,'Name', 'TFIDF, LSA, K-Medoids')

for i = 1:length(algorithms)
    
    % Specifying subplot values
    subplot(2,2,i)
    
    % Grouping the documents based on k-medoids 
    [groups,C] = kmedoids(mdl.DocumentScores, numTopics,...
                                'Distance'  , algorithms{i});

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentScores,'NumComponents',2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups)');
    
    % Title
    title(['K-Means ' upper(algorithms{i})])
    grid on

end

% Validation using SSE
[r, c] = size(full_M);
SSE = zeros(1,r);

for i = 1:r
    
    % Grouping the documents based on k-means 
    [groups,C]  = kmedoids(mdl.DocumentScores, i,...
                           'Distance'  , 'sqeuclidean');
                       
    % Obtaining the SSE
    tot = 0;
    for j = 1:length(mdl.DocumentScores)
        
        tot = tot + pdist([mdl.DocumentScores(j,:) ; C(groups(j),:)]);
        
    end

    % Storing the SSE Value
    SSE(i) = tot;
    
end

% Plotting the SSE
figure()
set(gcf,'Name', 'TFIDF, LSA, K-Medoids')
plot(1:r, SSE)

grid on
title('SSE vs. Number of Partitions')
xlabel('Number of Partitions')
ylabel('SSE')

%% Bag of Words, LDA, Hiearchal Linkage/Cluster

% Creating an LDA model from the bag of words
full_bag = full(bag.Counts);

numTopics = 3
mdl = fitlda(full_bag, numTopics)

figure()
set(gcf,'Name', 'Bag Of Words, LDA, Hiearchal Linkage/Cluster')

methods = {'single', 'complete', 'average', 'ward'};
metrics = 'euclidean';

for i = 1:length(methods)
    
    % Subplot Locations
    subplot(2,2,i)
    
    % Clustering the documents
    z = linkage(full_bag, methods{i}, metrics);
    groups = cluster(z, 'maxclust', numTopics);

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentTopicProbabilities, 'NumComponents', 2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups));
    
    % Creating the title
    title(['Hierarchal ' upper(methods{i}) ' Cluster'])
    grid on
    
end

figure()
set(gcf,'Name', 'Bag Of Words, LDA, Hiearchal Linkage/Cluster')
dendrogram(z)
title(['Dendrogram of ' upper(methods{i}) ' Cluster'])


%% Bag of Words, LSA, Hierarchal Linkage/Cluster

% Creating an LDA model from the bag of words
full_bag = full(bag.Counts);

numTopics = 3
mdl = fitlsa(full_bag, numTopics)

figure()
set(gcf,'Name', 'Bag Of Words, LSA, Hiearchal Linkage/Cluster')

methods = {'single', 'complete', 'average', 'ward'};
metrics = 'euclidean';

for i = 1:length(methods)
    
    % Subplot Locations
    subplot(2,2,i)
    
    % Clustering the documents
    z = linkage(full_bag, methods{i}, metrics);
    groups = cluster(z, 'maxclust', numTopics);

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentScores, 'NumComponents', 2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups));
    
    % Creating the title
    title(['Hierarchal ' upper(methods{i}) ' CLUSTER'])
    grid on
    
end

figure()
set(gcf,'Name', 'Bag Of Words, LSA, Hiearchal Linkage/Cluster')
dendrogram(z)
title(['Dendrogram of ' upper(methods{i}) ' Cluster'])


%% TFIDF, LDA, Hierarchal Linkage/Cluster

% Creating an LDA model from the bag of words
numTopics = 3
mdl = fitlda(round(full_M), numTopics)

figure()
set(gcf,'Name', 'TFIDF, LDA, Hiearchal Linkage/Cluster')

methods = {'single', 'complete', 'average', 'ward'};
metrics = 'euclidean';

for i = 1:length(methods)
    
    % Subplot Locations
    subplot(2,2,i)
    
    % Clustering the documents
    z = linkage(full_bag, methods{i}, metrics);
    groups = cluster(z, 'maxclust', numTopics);

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentTopicProbabilities, 'NumComponents', 2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups));
    
    % Creating the title
    title(['Hierarchal ' upper(methods{i}) ' Cluster'])
    grid on
    
end

figure()
set(gcf,'Name', 'TFIDF, LDA, Hiearchal Linkage/Cluster')
dendrogram(z)
title(['Dendrogram of ' upper(methods{i}) ' Cluster'])


%% TFIDF, LSA, Hierarchal Linkage/Cluster

numTopics = 3
mdl = fitlsa(full_M, numTopics)

figure()
set(gcf,'Name', 'TFIDF, LSA, Hiearchal Linkage/Cluster')

methods = {'single', 'complete', 'average', 'ward'};
metrics = 'euclidean';

for i = 1:length(methods)
    
    % Subplot Locations
    subplot(2,2,i)
    
    % Clustering the documents
    z = linkage(full_bag, methods{i}, metrics);
    groups = cluster(z, 'maxclust', numTopics);

    % Calculating PCA for 2 values only to visualize the plot
    [coeff,score] = pca(mdl.DocumentScores, 'NumComponents', 2);

    % Plotting Values
    colors = linspecer();
    scatter(score(:,1), score(:,2), [], colors(groups));
    
    % Creating the title
    title(['Hierarchal ' upper(methods{i}) ' Cluster'])
    grid on
    
end

figure()
set(gcf,'Name', 'TFIDF, LSA, Hiearchal Linkage/Cluster')
dendrogram(z)
title(['Dendrogram of ' upper(methods{i}) ' Cluster'])


%% TFIDF, DBSCAN

% DBSCAN referenced and used from:
% https://www.mathworks.com/matlabcentral/
%         fileexchange/52905-dbscan-clustering-algorithm

figure()
set(gcf,'Name', 'TFIDF, DBSCAN')
    
% Clustering the documents using DBSCAN
epsilon = 50;
MinPts  = 3;
IDX     = DBSCAN(full_M, epsilon, MinPts);

% Calculating PCA for 2 values only to visualize the plot
[coeff,score] = pca(full_M, 'NumComponents', 2);

% Plotting Values
colors = linspecer();

% Finding the maximum number of clusters
k = max(IDX);

% Looping through each cluster
for i=0:k
    
    % Obtaining the data for each cluster
    Xi = score(IDX==i,:);
    
    % Defining the point based on the cluster
    if i~=0
        
        Style = 'x';
        MarkerSize = 8;
        Color = colors(i,:);
        
    else
        
        Style = 'o';
        MarkerSize = 6;
        Color = [0 0 0];
        
    end
    
    if ~isempty(Xi)
        
        % Plotting the points for each cluster
        plot(Xi(:,1),Xi(:,2),Style,'MarkerSize',MarkerSize,'Color',Color);
        
    end
    
    hold on;
    
end

title(sprintf('DBSCAN Clustering (\\epsilon = %d, MinPts = %d)', epsilon, MinPts))
    
