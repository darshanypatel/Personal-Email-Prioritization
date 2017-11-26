function docs =  BatchPreProcess(documents)

% Remove all numbers
documents = documents.regexprep('([0-9]+)','');

% Remove the first set of stop words.
documents = documents.removeWords(stopWords);

% Remove the second set of stop words
stop_words_2 = {'forwarded', 'cc', 'bcc', 'subject', 'image'};
documents = documents.removeWords(stop_words_2);

docs = documents;

end