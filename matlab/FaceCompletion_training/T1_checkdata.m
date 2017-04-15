function datasize = T1_checkdata(data)
datacellnum = length(data);
for m = 1:datacellnum
    fprintf('Data name %d: %s.\n',m,data(m).name);
    fprintf('Data size %d: %d %d %d %d. \n',m,size(data(m).data));
    datasize{m} = size(data(m).data);
end
