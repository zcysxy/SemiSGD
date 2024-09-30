clc; clear all;
dim_arr = [2,5,8,10,20,25,40,50];
err_cor_mean_arr = zeros(1, length(dim_arr));
err_cor_std_arr = zeros(1, length(dim_arr));
err_lfa_mean_arr = zeros(1, length(dim_arr));
err_lfa_std_arr = zeros(1, length(dim_arr));

for i = 1:length(dim_arr)
	dim = dim_arr(i);
	main;
	err_cor_mean_arr(i) = err_cor_mean;
	err_cor_std_arr(i) = err_cor_std;
	err_lfa_mean_arr(i) = err_lfa_mean;
	err_lfa_std_arr(i) = err_lfa_std;
end

T = table(dim_arr', err_cor_mean_arr', err_cor_std_arr', err_lfa_mean_arr', err_lfa_std_arr', 'VariableNames', {'dim', 'err_cor_mean', 'err_cor_std', 'err_lfa_mean', 'err_lfa_std'});
writetable(T, 'err.csv');
