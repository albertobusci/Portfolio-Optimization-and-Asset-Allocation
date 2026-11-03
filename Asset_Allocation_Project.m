clear all
close all
clc
%% DATA IMPORT AND BASIC STATISTICS
path_map        = 'C:\Users\39348\OneDrive\Desktop\computational finance\progetto rota\';
filename_prices = 'asset_prices.csv';
filename_macro  = 'mapping_table.csv';
filename_cap    = 'capitalization_weights.csv';

% Import data as tables
table_prices = readtable(strcat(path_map, filename_prices));
macrogroup   = readtable(strcat(path_map, filename_macro));

% Macro groups: Cyclical, Defensive and Neutral views
idxCyclical  = strcmp(macrogroup.MacroGroup, 'Cyclical');
idxDefensive = strcmp(macrogroup.MacroGroup, 'Defensive');
idxNeutral   = strcmp(macrogroup.MacroGroup, 'Neutral');

% Transform prices to timetable
dates_all = table_prices{:,1};
prices_all = table_prices{:,2:end};
assetNamesCell = table_prices.Properties.VariableNames(2:end);
assetNames = string(assetNamesCell);
NumAssets  = numel(assetNames);

pricesTT = array2timetable(prices_all, 'RowTimes', dates_all, 'VariableNames', assetNamesCell); 

% Selection of the in-sample period (2018–2022)
start_dt = datetime('01/01/2018', 'InputFormat', 'dd/MM/yyyy'); 
end_dt   = datetime('31/12/2022', 'InputFormat', 'dd/MM/yyyy');

inSampleRange = timerange(start_dt, end_dt, 'closed'); 
inSampleTT    = pricesTT(inSampleRange,:);

prices_in  = inSampleTT.Variables;
dates_in   = inSampleTT.Time;

% Log-returns and basic statistics (daily, then annualized)
ret    = prices_in(2:end,:) ./ prices_in(1:end-1,:);
LogRet = log(ret);                      
T_in   = size(LogRet,1);

mu_daily   = mean(LogRet);             
sigma_daily = cov(LogRet);             

% Annualization (considering 252 trading days)
ExpRet = mu_daily * 252;               
V      = sigma_daily * 252;           

%% EXERCISE 1.a – Constrained Efficient Frontier (Portfolios A & B)
% Random portfolios generation
Nrand        = 100000;
RetRand      = zeros(1, Nrand);
VolaRand     = zeros(1, Nrand);
SharpeRand   = zeros(1, Nrand);
rf        = 0.0; % we don't have a risk factor                

for n = 1:Nrand
    w = rand(1, NumAssets);
    w = w ./ sum(w);                 
    RetRand(n)    = w * ExpRet';
    VolaRand(n)   = sqrt(w * V * w');
    SharpeRand(n) = (RetRand(n) - rf) / VolaRand(n);
end

% Plot: random portfolios cloud
figure;
scatter(VolaRand, RetRand, 8, SharpeRand, 'filled')
colorbar
colormap turbo
title('Random Portfolios: Expected Return vs Volatility')
xlabel('Volatility (annualized)')
ylabel('Expected Return (annualized)')
grid on; box on;

% Portfolio object with constraints
lb = zeros(NumAssets,1);
ub = 0.25 * ones(NumAssets,1);

p = Portfolio('AssetList', assetNamesCell);
p = setAssetMoments(p, ExpRet', V);
p = setBounds(p, lb, ub);
p = setBudget(p, 1, 1);


% Group constraints (Defensive ≤ 40%, Cyclical+Defensive between 45% and 75%)
vDef = double(idxDefensive(:))';
vCyc = double(idxCyclical(:))';
vCD  = vDef + vCyc;

% Using addGroups:
% 1) Defensive: 0 ≤ sum(Def) ≤ 0.40
% 2) Cyc+Def:   0 ≤ sum(CD) ≤ 0.75
% 3) Cyc+Def:   0.45 ≤ sum(CD) ≤ 1   (to enforce lower bound 45%)
G = [vDef; vCD; vCD];
LowerGroup = [0; 0; 0.45];
UpperGroup = [0.40; 0.75; 1.0];

p = addGroups(p, G, LowerGroup, UpperGroup);

% Efficient frontier under constraints
nPort = 100;
w_frontier = estimateFrontier(p, nPort);
[vol_frontier, ret_frontier] = estimatePortMoments(p, w_frontier);

Sharpe_frontier = (ret_frontier - rf) ./ vol_frontier;

% Minimum Variance Portfolio (Portfolio A)
[~, idxMVP_1a] = min(vol_frontier);
Portfolio_A = w_frontier(:, idxMVP_1a);

% Maximum Sharpe Ratio Portfolio (Portfolio B)
[~, idxMSRP_1a] = max(Sharpe_frontier);
Portfolio_B = w_frontier(:, idxMSRP_1a);

Ret_MSRP_1a  = Portfolio_B' * ExpRet';
Vola_MSRP_1a = sqrt(Portfolio_B' * V * Portfolio_B);
Sharpe_MSRP_1a = (Ret_MSRP_1a - rf) / Vola_MSRP_1a;

% Capital Market Line (using MSRP on constrained frontier)
CML_x = linspace(0, max(vol_frontier), 100);
CML_y = rf + ( (Ret_MSRP_1a - rf) / Vola_MSRP_1a ) .* CML_x;

% Plot Efficient Frontier + CML + Random Cloud
figure;
hold on;
scatter(VolaRand, RetRand, 8, [0.8 0.8 0.8], 'filled');
plot(vol_frontier, ret_frontier, 'b-', 'LineWidth', 2);
plot(CML_x, CML_y, 'r--', 'LineWidth', 1.8);
scatter(vol_frontier(idxMVP_1a),  ret_frontier(idxMVP_1a), 90, 'p', ...
    'MarkerFaceColor',[0.929 0.694 0.125], 'MarkerEdgeColor','k');
scatter(vol_frontier(idxMSRP_1a), ret_frontier(idxMSRP_1a), 90, 'p', ...
    'MarkerFaceColor',[0.494 0.184 0.556], 'MarkerEdgeColor','k');
xlabel('Volatility (annualized)');
ylabel('Expected Return (annualized)');
title('Exercise 1.a – Efficient Frontier with Constraints');
legend({'Random Portfolios','Efficient Frontier','CML', ...
       'MVP (A)','MSRP (B)'}, 'Location','southeast');
grid on; box on;

disp('Min Variance Portfolio weights (Portfolio A):');
disp(Portfolio_A);

disp('Max Sharpe Portfolio weights (Portfolio B):');
disp(Portfolio_B);

%% EXERCISE 1.b – Resampled Robust Frontier (Portfolios C & D)
% We are reusing the same constraints (p) but resampling (ExpRet, V) via simulated returns
Nsim  = 100;         
rf = 0; 

RiskSim   = zeros(nPort, Nsim);
RetSim    = zeros(nPort, Nsim);
SharpeSim = zeros(nPort, Nsim);
Weights   = zeros(NumAssets, nPort, Nsim);

rng(42);
for n = 1:Nsim
    R_sim = mvnrnd(mu_daily, sigma_daily, T_in);  
    e_i   = mean(R_sim);                         
    V_i   = cov(R_sim);                          
    P_sim = setAssetMoments(p, e_i' * 252, V_i * 252);
    w_sim = estimateFrontier(P_sim, nPort);
    Weights(:,:,n) = w_sim;
    [pf_risk, pf_ret] = estimatePortMoments(P_sim, w_sim);
    RiskSim(:, n) = pf_risk;
    RetSim(:,  n) = pf_ret;
    SharpeSim(:, n) = (pf_ret - rf) ./ pf_risk;
end

% Mean Frontier and mean Weights
meanRisk    = mean(RiskSim, 2);
meanRet     = mean(RetSim,  2);
meanWeights = mean(Weights, 3);

% Portfolio C: Robust MVP
[~, idxMVP_res]    = min(meanRisk);
Portfolio_C        = meanWeights(:, idxMVP_res);
mvp_risk_resampled = meanRisk(idxMVP_res);
mvp_ret_resampled  = meanRet(idxMVP_res);

% Portfolio D: Robust MSRP
meanSharpe = mean(SharpeSim, 2);
[~, idxMSRP_res] = max(meanSharpe);
Portfolio_D      = meanWeights(:, idxMSRP_res);
maxSharpe_risk   = meanRisk(idxMSRP_res);
maxSharpe_ret    = meanRet(idxMSRP_res);

% Comparison between the two different frontiers
figure;
hold on;
plot(vol_frontier, ret_frontier, 'Color',[0 0.447 0.741], 'LineWidth', 2);
plot(meanRisk,    meanRet,      'Color',[0.850 0.325 0.098], 'LineWidth', 2);
scatter(mvp_risk_resampled,   mvp_ret_resampled,   90, 'p', ...
    'MarkerFaceColor',[0.929 0.694 0.125], 'MarkerEdgeColor','k');
scatter(maxSharpe_risk, maxSharpe_ret, 90, 'p', ...
    'MarkerFaceColor',[0.494 0.184 0.556], 'MarkerEdgeColor','k');
xlabel('Volatility (annualized)');
ylabel('Expected Return (annualized)');
title('Exercise 1.b – Classical vs Resampled Efficient Frontier');
legend({'Classical Frontier','Resampled Frontier', ...
        'Robust MVP (C)','Robust MSRP (D)'}, ...
        'Location','southeast');
grid on; box on;

disp('Min Variance Portfolio weights using Robust method (Portfolio C):');
disp(Portfolio_C);

disp('Max Sharpe Portfolio weights using Robust Method (Portfolio D):');
disp(Portfolio_D);

%% EXERCISE 2 – Black–Litterman Model (Portfolios E & F)
cap_table = readtable(strcat(path_map, filename_cap));

% Market portfolio weights
wMKT = cap_table.MarketWeight(1:NumAssets);
wMKT = wMKT / sum(wMKT);
disp('Market-cap weighted portfolio:');
disp(wMKT);

lambda = 1.2;  % risk aversion coefficient
disp('Equilibrium Returns')
mu_eq  = lambda * V * wMKT % equilibrium returns
tau    = 1 / T_in; % uncertainty scaling factor

macro_BL = string(cap_table.MacroGroup(1:NumAssets));

% Introducing the views

% View 1: Cyclical assets outperform Defensive by +2% (annual)
P1 = zeros(1, NumAssets);
P1(macro_BL == "Cyclical")  =  1 / sum(macro_BL == "Cyclical");
P1(macro_BL == "Defensive") = -1 / sum(macro_BL == "Defensive");
q1 = 0.02;

% View 2: Asset 3 outperform Asset 11 by +1%
P2 = zeros(1, NumAssets);
P2(assetNames == "Asset3")  =  1;
P2(assetNames == "Asset11") = -1;
q2 = 0.01;

% View 3: Asset 7 outperform average Defensive by +0.5%
P3 = zeros(1, NumAssets);
P3(assetNames == "Asset7") = 1;
P3(macro_BL == "Defensive") = -1 / sum(macro_BL == "Defensive");
q3 = 0.005;

P = [P1; P2; P3];
q = [q1; q2; q3];

% Omega (view uncertainty)
Omega = diag(diag(P * (tau * V) * P'));

% Posterior Black–Litterman returns & covariance
A_BL    = inv(tau * V) + P' * (Omega \ P);
b_BL    = inv(tau * V) * mu_eq + P' * (Omega \ q);
cov_post = inv(A_BL);
disp('Posterior expected returns:');
mu_BL  = cov_post * b_BL
cov_BL = V+cov_post;

figure;
bar([mu_eq, mu_BL])
title('Equilibrium vs Black–Litterman Expected Returns')
xlabel('Assets')
ylabel('Annualized Return')
legend('Equilibrium', 'Black–Litterman', 'Location','best');
set(gca,'XTickLabel',assetNames,'XTickLabelRotation',45);

% Classical vs BL efficient frontier
pClassic = Portfolio('AssetList', assetNamesCell);
pClassic = setDefaultConstraints(pClassic);
pClassic = setAssetMoments(pClassic, ExpRet', V);
wClassic = estimateFrontier(pClassic, 100);
[volClassic, retClassic] = estimatePortMoments(pClassic, wClassic);

pBL = Portfolio('AssetList', assetNamesCell);
pBL = setDefaultConstraints(pBL);
pBL = setAssetMoments(pBL, mu_BL, cov_BL);
wBL = estimateFrontier(pBL, 100);
[volBL, retBL] = estimatePortMoments(pBL, wBL);

SharpeClassic = (retClassic - rf) ./ volClassic;
SharpeBL      = (retBL      - rf) ./ volBL;

[~, idxMSRP_classic] = max(SharpeClassic);
[~, idxMSRP_BL]      = max(SharpeBL);
[~, idxMVP_classic]  = min(volClassic);
[~, idxMVP_BL]       = min(volBL);

w_MVP_classic  = wClassic(:, idxMVP_classic);
w_MVP_BL       = wBL(:,      idxMVP_BL);
w_MSRP_classic = wClassic(:, idxMSRP_classic);
w_MSRP_BL      = wBL(:,      idxMSRP_BL);

% Portfolios E & F
disp('MVP under Black Litterman approach (Portfolio E):');
Portfolio_E = w_MVP_BL     % BL MVP
disp('MRSP under Black Litterman approach (Portfolio F):');
Portfolio_F = w_MSRP_BL    % BL Max Sharpe

% Plot efficient frontiers
figure;
hold on;
plot(volClassic, retClassic, 'b-', 'LineWidth', 2);
plot(volBL,      retBL,      'r-', 'LineWidth', 2);
scatter(volClassic(idxMVP_classic),  retClassic(idxMVP_classic),  80, 'b', 'filled');
scatter(volClassic(idxMSRP_classic), retClassic(idxMSRP_classic), 80, 'g', 'filled');
scatter(volBL(idxMVP_BL),            retBL(idxMVP_BL),            80, 'r', 'filled');
scatter(volBL(idxMSRP_BL),           retBL(idxMSRP_BL),           80, 'y', 'filled');
xlabel('Volatility (annualized)');
ylabel('Expected Return (annualized)');
title('Exercise 2 – Classical vs Black–Litterman Frontiers');
legend({'Classical Frontier','BL Frontier','MVP (Classic)','MSRP (Classic)','MVP (BL)','MSRP (BL)'}, 'Location','northwest');
grid on; box on;

% Compare weights: Classical vs BL MSRP
Tweights = table(assetNames', w_MSRP_classic, w_MSRP_BL,'VariableNames', ["Asset","Classical_MSRP","BL_MSRP"]);
disp(Tweights)

figure;
tiledlayout(1,2,"TileSpacing","compact");
nexttile;
bar(w_MSRP_classic);
title('Classical Max Sharpe Portfolio');
xticks(1:NumAssets); xticklabels(assetNames); xtickangle(45);

nexttile;
bar(w_MSRP_BL);
title('Black–Litterman Max Sharpe Portfolio');
xticks(1:NumAssets); xticklabels(assetNames); xtickangle(45);

% Pie charts: Market vs BL MVP 
figure;
tiledlayout(1,2,"TileSpacing","compact");
nexttile;
pie(wMKT);
title('Market Portfolio Weights');

nexttile;
pie(w_MVP_BL);
title('Black–Litterman MVP Weights');

lgd = legend(assetNames);
lgd.Layout.Tile = 'south';
lgd.NumColumns  = 4;

%% EXERCISE 3 – Diversification-Based Optimization (Portfolios G & H)
sigma_daily_i = sqrt(diag(sigma_daily));
w0 = ones(NumAssets,1) / NumAssets;

Aeq = ones(1,NumAssets);
beq = 1;
lb_DR  = zeros(NumAssets,1); 
ub_DR  = 0.2 * ones(NumAssets,1);

A = [];
b = [];

% Each macro group ≥ 15%
A = [A; -idxCyclical'];
b = [b; -0.15];
A = [A; -idxNeutral'];
b = [b; -0.15];
A = [A; -idxDefensive'];
b = [b; -0.15];

opts_fmin = optimoptions('fmincon','Display','none');

% Portfolio G: Max Diversification Ratio
funDR = @(w) - (w' * sigma_daily_i) / sqrt(w' * sigma_daily * w);
disp('Max Diversification Ratio Portfolio (Porfolio G):');
Portfolio_G = fmincon(funDR, w0, A, b, Aeq, beq, lb_DR, ub_DR, [], opts_fmin)

% Portfolio H: Equal Risk Contribution (ERC)
budget   = (1/NumAssets) * ones(NumAssets,1);
funERC   = @(w) sum(((w .* (sigma_daily * w)) / (w' * sigma_daily * w) - budget).^2);
disp('Equal Risk Contribution Portfolio (Portfolio H)');
Portfolio_H = fmincon(funERC, w0, A, b, Aeq, beq, lb_DR, ub_DR, [], opts_fmin)

% constraint check
if exist('check_constraints','file')
    check_constraints(Portfolio_G, lb_DR, ub_DR, idxDefensive, idxCyclical);
    check_constraints(Portfolio_H, lb_DR, ub_DR, idxDefensive, idxCyclical);
end

% Metrics vs Equally Weighted
disp('Equally Weighted Portfolio:');
EW = ones(NumAssets,1) / NumAssets

% Diversification Ratio
DR_G  = getDiversificationRatio(Portfolio_G, LogRet);
DR_H  = getDiversificationRatio(Portfolio_H, LogRet);
DR_EW = getDiversificationRatio(EW,        LogRet);

% Volatility
vol_G  = sqrt(Portfolio_G' * sigma_daily * Portfolio_G);
vol_H  = sqrt(Portfolio_H' * sigma_daily * Portfolio_H);
vol_EW = sqrt(EW'           * sigma_daily * EW);

% Sharpe
sh_G   = mean(LogRet * Portfolio_G) / vol_G;
sh_H   = mean(LogRet * Portfolio_H) / vol_H;
sh_EW  = mean(LogRet * EW)          / vol_EW;

% Effective number of assets
ENA_G  = 1 / sum(Portfolio_G.^2);
ENA_H  = 1 / sum(Portfolio_H.^2);
ENA_EW = 1 / sum(EW.^2);

MetricNames = ["Diversification Ratio"; "Volatility"; "Sharpe Ratio"; "Effective N Assets"];

MetricsTable = table([DR_G;  vol_G;  sh_G;  ENA_G], [DR_H;  vol_H;  sh_H;  ENA_H], [DR_EW; vol_EW; sh_EW; ENA_EW],'RowNames', MetricNames,'VariableNames', {'Portfolio_MDR', 'Portfolio_ERC', 'EquallyWeighted'});
disp(MetricsTable)

% DR vs Volatility scatter
figure; hold on;
scatter(vol_G,  DR_G,  120, 'r', 'filled');
scatter(vol_H,  DR_H,  120, 'b', 'filled');
scatter(vol_EW, DR_EW, 120, 'k', 'filled');
text(vol_G,  DR_G,  '  G: MaxDR','FontSize',10);
text(vol_H,  DR_H,  '  H: ERC','FontSize',10);
text(vol_EW, DR_EW, '  EW','FontSize',10);
xlabel('Volatility (daily)');
ylabel('Diversification Ratio');
title('Exercise 3 – Diversification Ratio vs Volatility');
grid on; box on;

% Macro-group exposures
w_macro_G  = [sum(Portfolio_G(idxCyclical)), sum(Portfolio_G(idxNeutral)), sum(Portfolio_G(idxDefensive))];
w_macro_H  = [sum(Portfolio_H(idxCyclical)), sum(Portfolio_H(idxNeutral)), sum(Portfolio_H(idxDefensive))];
w_macro_EW = [sum(EW(idxCyclical)),         sum(EW(idxNeutral)),         sum(EW(idxDefensive))];

figure;
bar([w_macro_G; w_macro_H; w_macro_EW])
xlabel('Portfolio');
ylabel('Allocation');
title('Exercise 3 – Macro-Group Exposures');
legend({'Cyclical','Neutral','Defensive'}, 'Location','northoutside','Orientation','horizontal');
set(gca,'XTickLabel',{'MaxDR (G)','ERC (H)','EW'});
grid on; box on;

%% EXERCISE 4 – PCA & CVaR (Portfolios I & J)
% PCA on CovSample
[coeff_full, score_full, latent, ~, explained] = pca(LogRet);

CumExplVar = cumsum(explained);
k = find(CumExplVar >= 85, 1);  % min number PC for ≥ 85% variance

% Covariance reconstruction from first k PC: Cov ≈ V_k * Λ_k * V_k'
V_k = coeff_full(:,1:k);
Lambda_k = diag(latent(1:k));
CovarPCA = V_k * Lambda_k * V_k';

% Plot variance explained
ExplainedVar = latent(1:k) / sum(latent);
figure;
bar(ExplainedVar*100);
title('Exercise 4 – Variance Explained by First k Principal Components');
xlabel('Principal Component');
ylabel('Explained Variance (%)');

n_PC = 1:length(explained);
figure;
plot(n_PC, CumExplVar, 'm-', 'LineWidth',1.8); hold on;
scatter(n_PC, CumExplVar,'m', 'filled');
grid on;
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('Exercise 4 – Cumulative Explained Variance');

% Portfolio I: Maximum Sharpe Ratio Portfolio under constraints 
x0 = ones(NumAssets,1) / NumAssets;
lb_PCA = zeros(1,NumAssets);
ub_PCA = 0.25 * ones(1,NumAssets);

v1 = coeff_full(:,1);  % first eigenvector

% Constraints + full investment + w'* v1 <= 0.5
A_PCA   = v1'; 
b_PCA   = 0.5;
Aeq_PCA = ones(1,NumAssets);
beq_PCA = 1;

% Sharpe (daily, rf=0) – invariance scale
func_sharpePCA = @(w) - ((mu_daily * w) / sqrt(w' * CovarPCA * w));

w_sharpe = fmincon(func_sharpePCA, x0, A_PCA, b_PCA, Aeq_PCA, beq_PCA,lb_PCA, ub_PCA);

disp('Maximum Sharpe Ratio Portfolio Under Set Constraints (Portfolio I):')
Portfolio_I = w_sharpe

% Portfolio J: Min CVaR (5%) with vol cap 15% annualized
alpha = 0.95;
annFactor = sqrt(252);
nonlinVolCap = @(w) deal( annFactor * sqrt(w' * CovarPCA * w) - 0.15 , [] );

% Rockafellar–Uryasev LP (on daily returns)
f_LP = [ zeros(NumAssets,1) ; 1 ; (1/((1-alpha)*T_in)) * ones(T_in,1) ];

A1 = [-LogRet , -ones(T_in,1) , -eye(T_in)];    % u_i >= loss - eta
A2 = [ zeros(T_in,NumAssets+1) , -eye(T_in) ];        % u_i >= 0
A_LP  = [A1 ; A2];
b_LP  = zeros(2*T_in,1);

Aeq_LP = [ones(1,NumAssets) , 0 , zeros(1,T_in)];
beq_LP = 1;

lb_LP = [ zeros(NumAssets,1) ; -Inf ; zeros(T_in,1) ];
ub_LP = [];

xLP = linprog(f_LP, A_LP, b_LP, Aeq_LP, beq_LP, lb_LP, ub_LP,optimoptions('linprog','Display','none'));
w_init = xLP(1:NumAssets);

% Nonlinear refinement with vol cap & box constraints
obj_CVaR = @(w) -mean( (LogRet*w) .* (LogRet*w < quantile(LogRet*w, 1-alpha)) );

lb2 = zeros(NumAssets,1);
ub2 = 0.25 * ones(NumAssets,1);
Aeq2 = ones(1,NumAssets);
beq2 = 1;

optionsNLP = optimoptions('fmincon','Display','iter','Algorithm','sqp');

w_CVaR = fmincon(obj_CVaR, w_init, [], [], Aeq2, beq2, lb2, ub2, nonlinVolCap, optionsNLP);

disp('By running the code we notice that it converged to an infeasible point so we test whether for our data the vol-cap constraint isnt too tight (no portfolio can go below 15% annualized vol-cap).');
p_test = Portfolio('AssetList', assetNamesCell);
p_test = setDefaultConstraints(p_test);
p_test = setBounds(p_test, zeros(NumAssets,1), 0.25*ones(NumAssets,1));
p_test = setAssetMoments(p_test, zeros(NumAssets,1), CovarPCA);

w_minVar_PCA = estimateFrontier(p_test,1);
vol_minVar_daily = sqrt(w_minVar_PCA' * CovarPCA * w_minVar_PCA);
disp('The lowest possible volatily reached is:');
vol_minVar_annual = annFactor * vol_minVar_daily

disp('Minimum Conditional Value-at-Risk at 5% level WITHOUT 15% vol-cap reached(Porfolio J):');
Portfolio_J= w_CVaR

% Performance comparison: Portfolio I vs J
pRet_I = LogRet * Portfolio_I;
pRet_J = LogRet * Portfolio_J;

% Equity curves (base 100)
equity_I = exp(cumsum(pRet_I));
equity_J = exp(cumsum(pRet_J));
equity_I = 100 * equity_I / equity_I(1);
equity_J = 100 * equity_J / equity_J(1);

[annRet_I, annVol_I, Sharpe_I, MaxDD_I, Calmar_I] = getPerformanceMetrics(equity_I);
[annRet_J, annVol_J, Sharpe_J, MaxDD_J, Calmar_J] = getPerformanceMetrics(equity_J);

% Tail risk (CVaR 5%)
VaR_I = quantile(pRet_I, 1-alpha);
VaR_J = quantile(pRet_J, 1-alpha);
CVaR_I = mean(pRet_I(pRet_I < VaR_I));
CVaR_J = mean(pRet_J(pRet_J < VaR_J));

perfTable_Portfolios = table([CVaR_I ; CVaR_J],[annVol_I ; annVol_J],[MaxDD_I ; MaxDD_J],'VariableNames', ["CVaR_5pct","AnnVol","MaxDD"],'RowNames', {'PCA MaxSharpe (I)','PCA CVaR (J)'});

disp('Exercise 4 – Portfolio Comparison (Tail Risk, Volatility, Max Drawdown)');
disp(perfTable_Portfolios);

figure;
plot(dates_in(2:end), equity_I, 'LineWidth',1.5); hold on;
plot(dates_in(2:end), equity_J, 'LineWidth',1.5);
datetick('x','yyyy');
xlabel('Date'); ylabel('Equity (base 100)');
title('Exercise 4 – Equity Curves: Portfolio I vs J');
legend('Portfolio I – PCA Max Sharpe', 'Portfolio J – CVaR + Vol Cap', ...
       'Location','best');
grid on; box on;

%% Summary Plots
% Portfolios A to J
PortfolioNames = ["A – MVP", ...
                  "B – Max Sharpe", ...
                  "C – Robust MVP", ...
                  "D – Robust Max Sharpe", ...
                  "E – BL MVP", ...
                  "F – BL Max Sharpe", ...
                  "G – Max Diversification", ...
                  "H – Equal Risk Contribution", ...
                  "I – PCA Max Sharpe", ...
                  "J – PCA CVaR"
                  ];

W_all = [Portfolio_A, Portfolio_B, ...
         Portfolio_C, Portfolio_D, ...
         Portfolio_E, Portfolio_F, ...
         Portfolio_G, Portfolio_H, ...
         Portfolio_I, Portfolio_J];

% TABLE OF WEIGHTS
SummaryTable = array2table(W_all',...
    'VariableNames', assetNamesCell, ...
    'RowNames', PortfolioNames);

disp("SUMMARY TABLE OF PORTFOLIO WEIGHTS (A–J)");
disp(SummaryTable);


% HEATMAP OF ALL PORTFOLIOS
figure;
heatmap(assetNames, PortfolioNames, W_all',...
        'Colormap', parula, ...
        'ColorLimits', [0 max(W_all(:))]);
title('Portfolio Weights – Heatmap (A–J)');
xlabel('Assets'); ylabel('Portfolio');


% MACRO-GROUP EXPOSURES BARPLOT (stacked)
w_macro = zeros(10,3);

for k = 1:10
    w = W_all(:,k);
    w_macro(k,:) = [
        sum(w(idxCyclical)), ...
        sum(w(idxNeutral)), ...
        sum(w(idxDefensive))
    ];
end

figure;
bar(w_macro, 'stacked');
colormap([0.8 0.2 0.2; 0.7 0.7 0.7; 0.2 0.4 0.8]);
legend({'Cyclical','Neutral','Defensive'}, ...
       'Location','northoutside','Orientation','horizontal');
set(gca,'XTickLabel',PortfolioNames,'XTickLabelRotation',45);
title('Macro-Group Exposures for Portfolios A–J');
ylabel('Weight Share');

%% Introduction to point 5: prices and return plots (16 ASSET, 2001-10-09 – 2024-11-29)
plot_start = datetime('09/10/2001','InputFormat','dd/MM/yyyy');
plot_end   = datetime('29/11/2024','InputFormat','dd/MM/yyyy');
plotRangeTT = pricesTT(timerange(plot_start, plot_end, 'closed'), :);
prices_plot = plotRangeTT.Variables;
dates_plot  = plotRangeTT.Time;

% Log-returns
ret_plot    = log(prices_plot(2:end,:) ./ prices_plot(1:end-1,:));
dates_ret   = dates_plot(2:end);

figure('Name','Prices 16 Asset','Position',[100 100 1200 800]);
tiledlayout(4,4,"TileSpacing","compact","Padding","compact");

for i = 1:NumAssets
    nexttile;
    plot(dates_plot, prices_plot(:,i), 'LineWidth', 1);
    title(assetNames(i), 'Interpreter','none','FontSize',9);
    grid on;
    if i <= 12
        set(gca,'XTickLabel',[]);
    else
        xlabel('Date');
    end
    ylabel('Price');
end

sgtitle('Assets historical prices (09/10/2001 – 29/11/2024)');

figure('Name','Rendimenti 16 Asset','Position',[200 150 1200 800]);
tiledlayout(4,4,"TileSpacing","compact","Padding","compact");

for i = 1:NumAssets
    nexttile;
    plot(dates_ret, ret_plot(:,i), 'LineWidth', 1);
    yline(0,'k-','LineWidth',0.5);
    title(assetNames(i), 'Interpreter','none','FontSize',9);
    grid on;  
    if i <= 12
        set(gca,'XTickLabel',[]);
    else
        xlabel('Date');
    end
    ylabel('Log-return');
end

sgtitle('Assets log-returns (09/10/2001 – 29/11/2024)');

%% EXERCISE 5 – Our own strategy: Defensive/Cyclical Moving Average BL Strategy
% Full-price matrix and dates
allDates   = pricesTT.Time;
prices_all = pricesTT.Variables;
LogRet_all = log(prices_all(2:end,:) ./ prices_all(1:end-1,:));
dates_ret_all = allDates(2:end);

% OOS window (2023–2024)
oos_start = datetime(2023,1,1);
oos_end   = datetime(2024,12,31);

% Rebalancing dates: first day of each month, then map to last available trading day before
rebal_dates = dateshift(oos_start,'start','month'):calmonths(1):dateshift(oos_end,'start','month');
rebal_trade_dates = NaT(size(rebal_dates));
for k = 1:numel(rebal_dates)
    idx = find(allDates <= rebal_dates(k) - days(1), 1, 'last');
    rebal_trade_dates(k) = allDates(idx);
end

% Moving averages (MA50 & MA200 on the prices)
MA50_all  = movmean(prices_all,  [49 0]);   
MA200_all = movmean(prices_all, [199 0]);  

% Storage for the results
weights_monthly = zeros(NumAssets, numel(rebal_trade_dates));
daily_ret_port  = zeros(length(LogRet_all),1);

% Portfolio template Black–Litterman
pBL_template = Portfolio('AssetList', assetNamesCell);
pBL_template = setDefaultConstraints(pBL_template);
pBL_template = setBounds(pBL_template, zeros(NumAssets,1), 0.15*ones(NumAssets,1));
pBL_template = setAssetMoments(pBL_template, zeros(NumAssets,1), V);

% Monthly rebalancing loop
for m = 1:numel(rebal_trade_dates)
    reb_date = rebal_trade_dates(m);
    idx_price = find(allDates == reb_date, 1);

    price_t = prices_all(idx_price, :);
    ma50_t  = MA50_all(idx_price, :);
    ma200_t = MA200_all(idx_price, :);

    % Compute MA signals
    pos_signal = (price_t > ma50_t) & (price_t > ma200_t);
    neg_signal = (price_t < ma50_t) & (price_t < ma200_t);
    nPos = sum(pos_signal);
    nNeg = sum(neg_signal);

    % Views containers
    q_t   = zeros(0,1);
    P_rows = zeros(0, NumAssets);

    if nPos > nNeg
        % Bullish regime: overweight cyclical
        for i = 1:NumAssets
            view = 0;
            if idxCyclical(i)
                view = 0.03;      % +3% tilting on cyclical
            end
            P_row = zeros(1, NumAssets);
            P_row(i) = 1;
            % relative-to-market view: asset i vs market portfolio
            P_row = P_row - wMKT(:)';
            P_rows = [P_rows; P_row];
            q_t   = [q_t;   view];
        end

    elseif nNeg > nPos
        % Bearish regime: overweight defensive
        for i = 1:NumAssets
            view = 0;
            if idxDefensive(i)
                view = 0.03;      % +3% tilting on defensive
            end
            P_row = zeros(1, NumAssets);
            P_row(i) = 1;
            P_row = P_row - wMKT(:)';
            P_rows = [P_rows; P_row];
            q_t   = [q_t;   view];
        end

    else
        % Neutral regime: no additional views
        mu_post  = mu_eq;
        cov_post = V;
    end

    % Black–Litterman posterior for this month
    if ~isempty(q_t)
        Omega   = diag(diag(P_rows * (tau * V) * P_rows'));
        A_BL    = inv(tau * V) + P_rows' * (Omega \ P_rows);
        b_BL    = inv(tau * V) * mu_eq + P_rows' * (Omega \ q_t);
        cov_post = inv(A_BL);
        mu_post  = cov_post * b_BL;
    end

    % Max–Sharpe portfolio (monthly)
    pBL = pBL_template;
    pBL = setAssetMoments(pBL, mu_post, cov_post);
    w_front = estimateFrontier(pBL, 100);
    [vols_f, rets_f] = estimatePortMoments(pBL, w_front);
    sharpe_f = (rets_f - 0) ./ vols_f;
    [~, idxMaxSharpe] = max(sharpe_f);
    w_star = w_front(:, idxMaxSharpe);

    weights_monthly(:, m) = w_star;
end

% Daily OOS portfolio returns (2023–2024)
idx_oos = find(dates_ret_all >= oos_start & dates_ret_all <= oos_end);
for ii = 1:length(idx_oos)
    dr_idx   = idx_oos(ii);
    cur_date = dates_ret_all(dr_idx);
    j = find(rebal_trade_dates <= cur_date, 1, 'last');
    if isempty(j), continue; end
    w_active = weights_monthly(:, j);
    daily_ret_port(dr_idx) = LogRet_all(dr_idx,:) * w_active;
end
daily_ret_oos = daily_ret_port(idx_oos);
dates_oos     = dates_ret_all(idx_oos);
equity        = 100 * cumprod(1 + daily_ret_oos);

% Performance metrics BL strategy
annRet  = mean(daily_ret_oos) * 252;
annVol  = std(daily_ret_oos) * sqrt(252);
Sharpe  = annRet / annVol;
cumret  = cumprod(1 + daily_ret_oos);
peak    = cummax(cumret);
drawdown = (cumret - peak) ./ peak;
MaxDD   = min(drawdown);
Calmar  = annRet / abs(MaxDD);

fprintf('OOS Performance Exercise 5 – BL MA Strategy:\n');
fprintf('AnnReturn: %.2f%%, AnnVol: %.2f%%, Sharpe: %.3f, MaxDD: %.2f%%, Calmar: %.3f\n',...
    100*annRet, 100*annVol, Sharpe, 100*abs(MaxDD), Calmar);

% Equity curve plot (BL strategy)
figure;
plot(dates_oos, equity, 'LineWidth', 1.6);
datetick('x','yyyy-mm','keepticks');
xlabel('Date'); ylabel('Equity (Base 100)');
title('Exercise 5 – BL Moving Average Strategy (OOS Equity)');
grid on; box on;

% Market-cap comparison
daily_ret_mkt = LogRet_all(idx_oos,:) * wMKT(:);
equity_mkt    = 100 * cumprod(1 + daily_ret_mkt);

annRet_mkt  = mean(daily_ret_mkt) * 252;
annVol_mkt  = std(daily_ret_mkt) * sqrt(252);
Sharpe_mkt  = annRet_mkt / annVol_mkt;
cumret_mkt  = cumprod(1 + daily_ret_mkt);
peak_mkt    = cummax(cumret_mkt);
drawdown_mkt = (cumret_mkt - peak_mkt) ./ peak_mkt;
MaxDD_mkt   = min(drawdown_mkt);
Calmar_mkt  = annRet_mkt / abs(MaxDD_mkt);

PerfTable = table( ...
    [annRet;      annRet_mkt], ...
    [annVol;      annVol_mkt], ...
    [Sharpe;      Sharpe_mkt], ...
    [abs(MaxDD);  abs(MaxDD_mkt)], ...
    [Calmar;      Calmar_mkt], ...
    'VariableNames', {'AnnReturn','AnnVol','Sharpe','MaxDrawdown','Calmar'}, ...
    'RowNames', {'BL_MA_Strategy','MarketCap'});

disp('Performance Comparison – Exercise 5');
disp(PerfTable);

% Equity comparison plot BL vs Market
figure;
plot(dates_oos, equity, 'LineWidth', 1.6); hold on;
plot(dates_oos, equity_mkt, '--', 'LineWidth', 1.6);
datetick('x','yyyy-mm','keepticks');
xlabel('Date'); ylabel('Equity (Base 100)');
title('Exercise 5 – BL MA Strategy vs Market-Cap');
legend('BL MA Strategy','Market-Cap','Location','best');
grid on; box on;

%% OOS PERFORMANCE OF PORTFOLIOS A–J (Jan 2023 – Dec 2024)
idx_oos  = find(dates_ret_all >= oos_start & dates_ret_all <= oos_end);
dates_oos = dates_ret_all(idx_oos);

PortfolioNames_OOS = { ...
    'A – MVP', ...
    'B – Max Sharpe', ...
    'C – Robust MVP', ...
    'D – Robust Max Sharpe', ...
    'E – BL MVP', ...
    'F – BL Max Sharpe', ...
    'G – Max Diversification', ...
    'H – Equal Risk Contribution', ...
    'I – PCA Max Sharpe', ...
    'J – PCA CVaR'};

W_list = { ...
    Portfolio_A, ...
    Portfolio_B, ...
    Portfolio_C, ...
    Portfolio_D, ...
    Portfolio_E, ...
    Portfolio_F, ...
    Portfolio_G, ...
    Portfolio_H, ...
    Portfolio_I, ...
    Portfolio_J};

nPortOOS = numel(W_list);

% Preallocazioni
daily_ret_OOS_all = zeros(numel(idx_oos), nPortOOS);
equity_OOS_all    = zeros(numel(idx_oos), nPortOOS);
annRet_OOS   = zeros(nPortOOS,1);
annVol_OOS   = zeros(nPortOOS,1);
Sharpe_OOS   = zeros(nPortOOS,1);
MaxDD_OOS    = zeros(nPortOOS,1);
Calmar_OOS   = zeros(nPortOOS,1);

for p = 1:nPortOOS
    w_p = W_list{p};
    r_p = LogRet_all(idx_oos,:) * w_p;
    daily_ret_OOS_all(:,p) = r_p;
    eq_p = 100 * cumprod(1 + r_p);
    equity_OOS_all(:,p) = eq_p;
    
    % Performance metrics OOS
    annRet_OOS(p) = mean(r_p) * 252;
    annVol_OOS(p) = std(r_p)  * sqrt(252);
    Sharpe_OOS(p) = annRet_OOS(p) / annVol_OOS(p);
    
    cumret  = cumprod(1 + r_p);
    peak    = cummax(cumret);
    drawdown = (cumret - peak) ./ peak;
    MaxDD_OOS(p)  = min(drawdown);
    Calmar_OOS(p) = annRet_OOS(p) / abs(MaxDD_OOS(p));
end

% Summary Table
PerfTable_OOS = table( ...
    annRet_OOS, ...
    annVol_OOS, ...
    Sharpe_OOS, ...
    abs(MaxDD_OOS), ...
    Calmar_OOS, ...
    'VariableNames', {'AnnReturn','AnnVol','Sharpe','MaxDrawdown','Calmar'}, ...
    'RowNames', PortfolioNames_OOS);

disp('OUT-OF-SAMPLE PERFORMANCE (Jan 2023 – Dec 2024) – Portfolios A–J');
disp(PerfTable_OOS);

% Equity curves of A–J
figure;
hold on;
colors = lines(nPortOOS);

for p = 1:nPortOOS
    plot(dates_oos, equity_OOS_all(:,p), 'LineWidth', 1.5, 'Color', colors(p,:));
end

datetick('x','yyyy-mm','keepticks');
xlabel('Date');
ylabel('Equity (Base 100)');
title('Out-of-sample Equity Curves (2023–2024) – Portfolios A–J');
legend(PortfolioNames_OOS, 'Location','bestoutside');
grid on; box on;

exportgraphics(gcf, 'OOS.pdf')

%% EXERCISE 5.d – 2025 OOS Extension (Final Check)

% Importing new data
filename_prices_2025 = 'asset_prices_out_of_sample.csv';
table_prices_2025 = readtable(strcat(path_map, filename_prices_2025));
dates_2025 = table_prices_2025{:,1};
prices_2025_new = table_prices_2025{:,2:end};
pricesTT_2025 = array2timetable(prices_2025_new, 'RowTimes', dates_2025, 'VariableNames', assetNamesCell);

% Combining data: full PricesTT (up to 2024) + PricesTT_2025
fullPricesTT = [pricesTT; pricesTT_2025];
fullPricesTT = unique(fullPricesTT, 'rows', 'stable'); 
disp(['Data updated to: ', datestr(fullPricesTT.Time(end))]);

% Redefining dates for the full period (2001-End 2025)
allDates    = fullPricesTT.Time;
prices_all  = fullPricesTT.Variables;
LogRet_all  = log(prices_all(2:end,:) ./ prices_all(1:end-1,:));
dates_ret_all = allDates(2:end);

% OOS window 2025
oos_start = datetime(2024,12,02);
oos_end   = allDates(end);

% Rebalancing dates: first day of each month, then map to last available trading day before
rebal_dates = dateshift(oos_start,'start','month'):calmonths(1):dateshift(oos_end,'start','month');
rebal_trade_dates = NaT(size(rebal_dates));
for k = 1:numel(rebal_dates)
    idx = find(allDates <= rebal_dates(k) - days(1), 1, 'last');
    rebal_trade_dates(k) = allDates(idx);
end

% Moving averages (MA50 & MA200 on the prices)
MA50_all  = movmean(prices_all,  [49 0]);   
MA200_all = movmean(prices_all, [199 0]);  

% Storage for the results
weights_monthly = zeros(NumAssets, numel(rebal_trade_dates));
daily_ret_port  = zeros(length(LogRet_all),1);

% Portfolio template Black–Litterman
pBL_template = Portfolio('AssetList', assetNamesCell);
pBL_template = setDefaultConstraints(pBL_template);
pBL_template = setBounds(pBL_template, zeros(NumAssets,1), 0.15*ones(NumAssets,1));
pBL_template = setAssetMoments(pBL_template, zeros(NumAssets,1), V);

% Monthly rebalancing loop
for m = 1:numel(rebal_trade_dates)
    reb_date = rebal_trade_dates(m);
    idx_price = find(allDates == reb_date, 1);

    price_t = prices_all(idx_price, :);
    ma50_t  = MA50_all(idx_price, :);
    ma200_t = MA200_all(idx_price, :);

    % Compute MA signals
    pos_signal = (price_t > ma50_t) & (price_t > ma200_t);
    neg_signal = (price_t < ma50_t) & (price_t < ma200_t);
    nPos = sum(pos_signal);
    nNeg = sum(neg_signal);

    % Views containers
    q_t   = zeros(0,1);
    P_rows = zeros(0, NumAssets);

    if nPos > nNeg
        % Bullish regime: overweight cyclical
        for i = 1:NumAssets
            view = 0;
            if idxCyclical(i)
                view = 0.03;      % +3% tilting on cyclical
            end
            P_row = zeros(1, NumAssets);
            P_row(i) = 1;
            % relative-to-market view: asset i vs market portfolio
            P_row = P_row - wMKT(:)';
            P_rows = [P_rows; P_row];
            q_t   = [q_t;   view];
        end

    elseif nNeg > nPos
        % Bearish regime: overweight defensive
        for i = 1:NumAssets
            view = 0;
            if idxDefensive(i)
                view = 0.03;      % +3% tilting on defensive
            end
            P_row = zeros(1, NumAssets);
            P_row(i) = 1;
            P_row = P_row - wMKT(:)';
            P_rows = [P_rows; P_row];
            q_t   = [q_t;   view];
        end

    else
        % Neutral regime: no additional views
        mu_post  = mu_eq;
        cov_post = V;
    end

    % Black–Litterman posterior for this month
    if ~isempty(q_t)
        Omega   = diag(diag(P_rows * (tau * V) * P_rows'));
        A_BL    = inv(tau * V) + P_rows' * (Omega \ P_rows);
        b_BL    = inv(tau * V) * mu_eq + P_rows' * (Omega \ q_t);
        cov_post = inv(A_BL);
        mu_post  = cov_post * b_BL;
    end

    % Max–Sharpe portfolio (monthly)
    pBL = pBL_template;
    pBL = setAssetMoments(pBL, mu_post, cov_post);
    w_front = estimateFrontier(pBL, 100);
    [vols_f, rets_f] = estimatePortMoments(pBL, w_front);
    sharpe_f = (rets_f - 0) ./ vols_f;
    [~, idxMaxSharpe] = max(sharpe_f);
    w_star = w_front(:, idxMaxSharpe);

    weights_monthly(:, m) = w_star;
end

% Daily OOS portfolio returns (2023–2024)
idx_oos = find(dates_ret_all >= oos_start & dates_ret_all <= oos_end);
for ii = 1:length(idx_oos)
    dr_idx   = idx_oos(ii);
    cur_date = dates_ret_all(dr_idx);
    j = find(rebal_trade_dates <= cur_date, 1, 'last');
    if isempty(j), continue; end
    w_active = weights_monthly(:, j);
    daily_ret_port(dr_idx) = LogRet_all(dr_idx,:) * w_active;
end
daily_ret_oos = daily_ret_port(idx_oos);
dates_oos     = dates_ret_all(idx_oos);
equity        = 100 * cumprod(1 + daily_ret_oos);

% Performance metrics BL strategy
annRet  = mean(daily_ret_oos) * 252;
annVol  = std(daily_ret_oos) * sqrt(252);
Sharpe  = annRet / annVol;
cumret  = cumprod(1 + daily_ret_oos);
peak    = cummax(cumret);
drawdown = (cumret - peak) ./ peak;
MaxDD   = min(drawdown);
Calmar  = annRet / abs(MaxDD);

fprintf('OOS Performance Exercise 5 – BL MA Strategy:\n');
fprintf('AnnReturn: %.2f%%, AnnVol: %.2f%%, Sharpe: %.3f, MaxDD: %.2f%%, Calmar: %.3f\n',...
    100*annRet, 100*annVol, Sharpe, 100*abs(MaxDD), Calmar);

% Equity curve plot (BL strategy)
figure;
plot(dates_oos, equity, 'LineWidth', 1.6);
datetick('x','yyyy-mm','keepticks');
xlabel('Date'); ylabel('Equity (Base 100)');
title('Exercise 5 – BL Moving Average Strategy (OOS Equity)');
grid on; box on;

% Market-cap comparison
daily_ret_mkt = LogRet_all(idx_oos,:) * wMKT(:);
equity_mkt    = 100 * cumprod(1 + daily_ret_mkt);

annRet_mkt  = mean(daily_ret_mkt) * 252;
annVol_mkt  = std(daily_ret_mkt) * sqrt(252);
Sharpe_mkt  = annRet_mkt / annVol_mkt;
cumret_mkt  = cumprod(1 + daily_ret_mkt);
peak_mkt    = cummax(cumret_mkt);
drawdown_mkt = (cumret_mkt - peak_mkt) ./ peak_mkt;
MaxDD_mkt   = min(drawdown_mkt);
Calmar_mkt  = annRet_mkt / abs(MaxDD_mkt);

PerfTable = table( ...
    [annRet;      annRet_mkt], ...
    [annVol;      annVol_mkt], ...
    [Sharpe;      Sharpe_mkt], ...
    [abs(MaxDD);  abs(MaxDD_mkt)], ...
    [Calmar;      Calmar_mkt], ...
    'VariableNames', {'AnnReturn','AnnVol','Sharpe','MaxDrawdown','Calmar'}, ...
    'RowNames', {'BL_MA_Strategy','MarketCap'});

disp('Performance Comparison – Exercise 5');
disp(PerfTable);

% Equity comparison plot BL vs Market
figure;
plot(dates_oos, equity, 'LineWidth', 1.6); hold on;
plot(dates_oos, equity_mkt, '--', 'LineWidth', 1.6);
datetick('x','yyyy-mm','keepticks');
xlabel('Date'); ylabel('Equity (Base 100)');
title('Exercise 5 – BL MA Strategy vs Market-Cap');
legend('BL MA Strategy','Market-Cap','Location','best');
grid on; box on;