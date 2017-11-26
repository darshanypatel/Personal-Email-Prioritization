clc;

% Defining files to parse
files = dir('*.csv');

% Iterating through files
for i = 1:length(files)
    
    % Setting the filename
    filename = files(i).name
    
    % Opening the filename to read from
    FID = fopen(filename, 'r');
    
    % Opening file to write to
    % FID_write = fopen(['tmp_' filename], 'w');
    
    % Iterate through the CSV file
    while ~feof(FID)

        line = fgetl(FID);
        tok = split(line,',');
        
        body = tok{7};
        
        [s1] = strfind(body, 'X-From:');
        [s2] = strfind(body, 'X-To:');
        [s3] = strfind(body, 'X-Folder');
        
        from = body(s1(1)+7:s2(1)-1);
        to = body(s2(1):s3(1));
        
        s4 = strfind(to, 'X-cc:');
        s5 = strfind(to, 'X-bcc:');
        
        to_to  = to(1:s4(1)-1);
        to_cc  = to(s4(1):s5(1)-1);
        to_bcc = to(s5(1):end);
        
        % tok{5}
   
        
        % If the body is empty, ignore it
%         if isempty(body(e-40:e+1))
%             continue
%         end
%         
%         % Storing the tokens
%         a = tok{1};
        
        % Strip the first part of the body
        % fprintf(FID_write, '%s,%s,%s,%s,%s\n', a{2}, a{3}, a{4}, a{5}, strip(body(e+1:end)));
        
    end
    
    % Close files
    fclose(FID);
    % fclose(FID_write);
    
end