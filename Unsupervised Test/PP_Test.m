
FID = fopen('words.txt','r');
words = {};
i = 1;

while ~feof(FID)
    words{i,1} = fgetl(FID);
    i = i+1;
end

fclose('all')

%%

b = a.regexprep('([0-9]+)','')
c = b;

%%

d = b.doc2cell()
f = d{1};
tic
for i = 1:length(d{1})
    if sum( strcmpi( d{1}{i} , words )) == 0
        c = c.removeWords(d{1}{i});
    end
end
toc

%%

stopWords2 = {'forwarded', 'cc', 'bcc', 'subject'};

e = c.removeWords(stopWords2)