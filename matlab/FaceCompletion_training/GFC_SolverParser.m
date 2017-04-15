function Solver = VGG_SolverParser(solver_def_file)
if ~exist('solver_def_file','var')||isempty(solver_def_file)
    error('You need a solver definition file');
end
if ~exist('Solver','var')
    fprintf('Creating a VGG pretrained solver structure.');
end
Solver = [];
fout = fopen(solver_def_file,'r');
tline = fgetl(fout);
while ischar(tline)
    disp(tline)
    ind = find(tline == '"',1);
    if  ~isempty(ind)
        field = tline(ind + 1 : end - 1);
        ind2 = find(tline == ':',1);
        name = tline(1:ind2-1);
    else
        ind2 = find(tline == ':',1);
        if isempty(ind2)
            error('incorrect format.')
        end
        ctr = tline(ind2+2:end);
        if isempty(str2num(ctr))
            field = ctr;
        else
            field = str2double(ctr);
        end
        name = tline(1:ind2-1);
    end
    Solver = setfield(Solver, name, field);
    tline = fgetl(fout);
end
fclose(fout);
if ~isfield(Solver, 'solver_mode')
    Solver.solver_mode = 'GPU';
end
if ~isfield(Solver, 'device_id') && strcmp(Solver.solver_mode, 'GPU')
        Solver.device_id = 0;
end
