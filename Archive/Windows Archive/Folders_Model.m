cd('C:\Users\Alper Ender\Desktop\CSC522Project\raw_data_folders\allen-p\inbox')

files = dir;

docs = {};
count = 1;

for i = 1:length(files)
   name = files(i).name;
   if sum(strcmpi(name,{'.','..','.DS_STORE'}))==0
       disp(name)
       docs{count,1} = Folders_PreProcess(fileread(name))
       count = count + 1;
   end
    
end