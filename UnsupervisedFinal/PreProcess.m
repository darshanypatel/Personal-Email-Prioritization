%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% @AutoGenerated
%
% Filename: PreProcess.m
% Author: Alper Ender
% Date: November 2017
% Description: Preprocess the individual documents
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\

function [doc] = PreProcess(text)

% Erase punctuation.
cleanTextData = erasePunctuation(text);

% Convert the text data to lowercase.
cleanTextData = lower(cleanTextData);

% Tokenize the text.
documents = tokenizedDocument(cleanTextData);

% Remove words with 2 or fewer characters, and words with 15 or greater
% characters.
% documents = removeShortWords(documents,2)
% documents = removeLongWords(documents,15);

% Normalize the words using the Porter stemmer.
% doc = normalizeWords(documents);

doc = documents;

end