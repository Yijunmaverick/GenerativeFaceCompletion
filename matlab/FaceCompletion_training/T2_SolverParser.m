function [ Solver ] = T2_SolverParser( solver_def_file, resume_file )
%   parse solver file, or merge with resume_file
if ~exist('solver_def_file','var')||isempty(solver_def_file)
    error('You need a solver definition file');
end
fout = fopen(solver_def_file,'r');
tline = fgetl(fout);
if ~exist('resume_file','var')||isempty(resume_file)
    Solver = [];
else
    load(resume_file);
end
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
if isfield(Solver,'model')
    lnum = length(Solver.model);
    for ind = 1:lnum
        if strcmp(Solver.model(ind).layer_names,sprintf('fc%d',ind))
            Solver.model(ind).layer_names = sprintf('conv%d',ind);
            weights = Solver.model(ind).weights{1};
            [s1,s2] = size(weights);
            [~,~,~,ch] = size(Solver.model(ind-1).weights{1});
            filtersize = sqrt(s1/ch);
            weights = reshape(weights,[filtersize,filtersize,ch,s2]);
            Solver.model(ind).weights{1} = weights;
        end
    end
end
if strcmp(Solver.solver_mode,'GPU') && ~isfield(Solver, 'device_id')
    Solver.device_id = 0;
end
end

