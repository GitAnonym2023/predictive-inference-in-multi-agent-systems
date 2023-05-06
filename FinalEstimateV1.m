function [mu, B_kt, Acq_flag]=FinalEstimateV1(ksi, Loc_data, Acq_data, B_kt, k, L, delta, Sigma_e)
    Acq_flag = zeros(size(ksi));              % For plotting purposes
    mu = nan(size(ksi));

    for ii = 1:length(ksi)
       
        [y_bar, B_opt,h] = NadarayaWatsonV7(ksi(ii),Loc_data(k,:,1),Loc_data(k,:,2),NaN,L,delta,Sigma_e(k)); 
        
        I = [];
        if ~isempty(Acq_data{k}), I = find(L*abs(ksi(ii)-Acq_data{k}(:,1))+Acq_data{k}(:,3)<B_opt);end
        if isempty(I)
           mu(ii) = y_bar;
           B_kt(k,ii) = B_opt;

           Acq_flag(ii) = 0;
        else
           Acq_meas_I = Acq_data{k}(I,:);
           [B_kt(k,ii),l] = min(L*abs(ksi(ii)-Acq_meas_I(:,1))+Acq_meas_I(:,3));
           mu(ii) = Acq_meas_I(l,2);

           Acq_flag(ii) = 1;
        end
    end