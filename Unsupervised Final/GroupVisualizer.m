

% Good ones: 3,5,8,9,12,15,16,18,19,20
k = 18

% for k = 1:20

b = docs(groups==k);
e = string();

for i = 1:length(b)
    c = b(i).string;
    d = '';
    for j = 1:length(c)
        d = [d ' ' char(c(j))];
    end
    e(i,1) = d;
end

figure
wordcloud(e)
title(sprintf('Group: %d',k))

% end
