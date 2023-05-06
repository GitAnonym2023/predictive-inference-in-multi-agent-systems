function y = new_output (ksi, sigma_e,m,seed)
    %rng(seed);
    y = m(ksi) + normrnd(0,sigma_e,1,1);   
end