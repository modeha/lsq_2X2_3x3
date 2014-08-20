function printResults()
load('pdco','pdco');
    Number_of_Iteration_pdco = pdco.get('Number of iteration');
    Name_pdco = pdco.get('Name of problem');
    Time_pdco = pdco.get('Time');
    Objective_pdco = pdco.get('Objective');
    nnzmnimizer_pdco = pdco.get('Number of non zero in minimizer');
    nnzoriginal_pdco = pdco.get('Number of non zero in original');
    gap_pdco =pdco.get('gap');
    
load('l1ls','l1ls');

    Number_of_Iteration_l1ls = l1ls.get('Number of iteration');
    Name_l1ls  = l1ls.get('Name of problem');
    Time_l1ls = l1ls.get('Time');
    Objective_l1ls = l1ls.get('Objective');
    nnzmnimizer_l1ls = l1ls.get('Number of non zero in minimizer');
    nnzoriginal_l1ls = l1ls.get('Number of non zero in original');
    gap_l1ls = l1ls.get('complementary');
fprintf('name iter \t obj \t\t gap \t\t time \t\t nnz_original \t nnz_minimizer\n');
dim = size(Time_pdco);
for i=1:dim(1)

fprintf('%-4d %-4d\t%-.6e\t%-.6e\t%-.6e\t%-.6e\t %-.6e %5s\n',...
    Name_pdco(i),Number_of_Iteration_pdco(i),gap_pdco(i),Objective_pdco(i),...
    Time_pdco(i), nnzoriginal_pdco(i), nnzmnimizer_pdco(i),'   PDCO');
fprintf('\n');
fprintf('%-4d %-4d\t%-.6e\t%-.6e\t%-.6e\t%-.6e\t %-.6e %5s \n',...
    Name_l1ls(i),Number_of_Iteration_l1ls(i),gap_l1ls(i),Objective_l1ls(i),...
    Time_l1ls(i), nnzoriginal_l1ls(i), nnzmnimizer_l1ls(i),'   l1ls');
fprintf('\n');
end
end

%en=pdco.keys();

% while (en.hasMoreElements())
%     java.lang.System.out.println(en.nextElement())
%     pdco.get(en.nextElement())
% end
% keyboard
