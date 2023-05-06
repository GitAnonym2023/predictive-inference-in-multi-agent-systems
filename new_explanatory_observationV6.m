function ksi = new_explanatory_observationV6(k, D, M,seed, state)
    persistent mu, 
    persistent sig;

    if strcmp(state, 'Initialize')
        rng(seed);
        mu = repmat(D(1),M,1) + D(2)*rand(M,1);
        sig = 0.1*ones(M,1) + 0.5*rand(M,1);
    else
        ksi = inf;
        while (ksi<D(1))||(ksi>D(2))        % Keep explanatory data within domain D
%          ksi = 2*rand(1)+k-1;      
           ksi = normrnd(mu(k),sig(k),1,1);
        end
    end
end