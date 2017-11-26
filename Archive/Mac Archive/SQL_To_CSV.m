clc; clear; close all;

%%

% STARTING LOCATION = 200
% ENDING LOCATION = 253487 - 200 - 529 = 252758

data = cell(252758,6);
FID = fopen('enron-mysqldump_v5.sql','r');

for i = 1:200
   fgetl(FID); 
end

% for i = 1:253287
%     a = fgetl(FID);
% %     if contains(a,'/*!40000 ALTER TABLE `message` ENABLE KEYS */;')
% %         break;
% %     end
%     fprintf('%s\n\n',a)
% end

count = 1;
for i = 1:253287
    
    rawIn = fgetl(FID);
    
    if contains(rawIn, 'INSERT INTO `message`')
        continue
    end
    
    [tokens, start, endloc] = regexp(rawIn,'([0-9]*, ''.*'', ''.*'', ''.*'', ''.*'', ''.*'')','tokens');
    entry = tokens{1}{1};
    [start, endloc] = regexp(entry,'[0-9]*,','once');
    number = entry(start:endloc-1);
    entry(start:endloc+2) = [];
    [start,endloc ]= regexp(entry ,''', ''','once');
    sender = entry(1:start-1);
    entry(1:endloc) = [];
    [start,endloc ]= regexp(entry ,''', ''','once');
    dateT = entry(1:start-1);
    entry(1:endloc) = [];
    [start,endloc ]= regexp(entry ,''', ''','once');
    ID = entry(1:start-1);
    entry(1:endloc) = [];
    [start,endloc ]= regexp(entry ,''', ''','once');
    Subject = entry(1:start-1);
    entry(1:endloc) = [];
    [start,endloc ]= regexp(entry ,''', ''.*''' ,'once');
    foldername = entry(start+4:endloc-1);
    entry(start:endloc) = [];
    message = entry;
    
    data{count, 1} = str2double(number);
    data{count, 2} = sender;
    data{count, 3} = dateT;
    data{count, 4} = ID;
    data{count, 5} = Subject;
    data{count, 6} = foldername;
    data{count, 7} = message;
    
    count = count + 1;
    disp(count)
    
end

close('all')
