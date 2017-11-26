fid = fopen('Enron_Emails_Uncleaned.csv');
email_counter = 1;
num_words = [];


while ~feof(fid)
    line = fgetl(fid);
    [b,e] = regexp(line, '(X-FileName:)');
    line(1:e) = [];
    num_words(email_counter) = length(find(~cellfun(@isempty,split(line, ' '))));
    email_counter = email_counter + 1;
    if mod(email_counter,10000) == 0
        disp(email_counter)
    end
end

email_counter

fclose(fid)