function [annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(equity)

    % Ensure column vector
    equity = equity(:);

    % Compute daily returns
    ret = [0; diff(equity) ./ equity(1:end-1)];

    % Number of trading days per year
    periodsPerYear = 252;

    % Annualized return
    cumulativeReturn = equity(end) / equity(1) - 1;
    annRet = (1 + cumulativeReturn)^(periodsPerYear / length(ret)) - 1;

    % Annualized volatility
    annVol = std(ret) * sqrt(periodsPerYear);

    % Sharpe ratio (risk-free = 0)
    Sharpe = annRet / annVol;

    % Max Drawdown
    runningMax = cummax(equity);
    dd = (equity - runningMax) ./ runningMax;
    MaxDD = min(dd);     % negative number

    % Calmar ratio
    Calmar = annRet / abs(MaxDD);

end