% Constraints check function
function check_constraints(w, lb, ub, Def, Cyc)
    NumAssets = length(w);
    if any(w < lb - 1e-6) || any(w > ub + 1e-6)
        warning('Some weights do NOT respect upper or lower bounds.');
    else
        disp('ALL weights respect upper and lower bounds.');
    end
    if abs(sum(w) - 1) > 1e-6
        warning('The sum of the weights is NOT 1.');
    else
        disp('The sum of weights IS 1.');
    end

    vDef = double(Def(:))';
    vCyc = double(Cyc(:))';
    vCD  = vDef + vCyc;

    sumDef = sum(w .* vDef);
    sumCD  = sum(w .* vCD);

    if sumDef > 0.40 + 1e-6
        warning('Defensive Constraint <= 0.40 VIOLATED (%.4f)', sumDef);
    else
        disp('Defensive Constraint <= 0.40 RESPECTED');
    end

    if sumCD > 0.75 + 1e-6
        warning(['Cyclical Constraint <= 0.75 VIOLATED (%.4f)', sumCD]);
    else
        disp('Cyclical + Defensive Constraint <= 0.75 RESPECTED');
    end

    if sumCD < 0.45 - 1e-6
        warning('Cyclical + Defensive Constraint >= 0.45 VIOLATED (%.4f)', sumCD);
    else
        disp('Cyclical + Defensive >= 0.45 RESPECTED');
    end
end
