function select_ksi_request = select_arg2requestV6(Local, Acquired, share_type,tau)
    % share_type: 'urand', 'worst_B', 'biggest_gap','worst_B_rand'
    global D;  %#ok<GVMIS>                          % The total inference domain (D in a paper)  
    global L;  %#ok<GVMIS>                          % A known Lipschitz constant 
    Local = Local(:,[1,2,4,5]);                       % Do not share raw output meas.
    L_ind = not(isnan(sum(Local,2)));               % The number of well defined local tuples
    A_ind = not(isnan(sum(Acquired,2)));            % The number of well defined acquired tuples
    Loc_Acq_tuples = sortrows([Local(L_ind,:);Acquired(A_ind,:)]);
    Tot_num = size(Loc_Acq_tuples,1);
    %rng(seed);
    if Tot_num == 0
        select_ksi_request = NaN;
        return
    end
    if strcmp(share_type,'worst_B_rand') 
        ksi = Loc_Acq_tuples(:,1);
        B = Loc_Acq_tuples(:,3);
        ksi_centers = (B(2:end)-B(1:end-1)+L*(ksi(1:end-1)+ksi(2:end)))/(2*L); %  % see e.g. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        B_local_max = L*(ksi_centers-ksi(1:end-1))+B(1:end-1);
        B_left = -L*(D(1)-ksi(1))+B(1);
        B_right = L*(D(2)-ksi(end))+B(end);
        B_set = [B_left; B_local_max; B_right];
        ksi_distr = [D(1);ksi_centers;D(2)];
        B_distr = exp(-tau*B_set);
        B_distr = B_distr/norm(B_distr,1);
        B_CDF = cumsum(B_distr);
        ind = find(B_CDF<rand,1,'last');
        if isempty(ind)
            ind = 1;
        else 
            ind = ind + 1;
        end
        select_ksi_request = ksi_distr(ind);         
    elseif strcmp(share_type,'worst_B') 
        ksi = Loc_Acq_tuples(:,1);
        B = Loc_Acq_tuples(:,3);
        ksi_centers = (B(2:end)-B(1:end-1)+L*(ksi(1:end-1)+ksi(2:end)))/(2*L); %  % see e.g. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
        B_local_max = L*(ksi_centers-ksi(1:end-1))+B(1:end-1);
        B_left = -L*(D(1)-ksi(1))+B(1);
        B_right = L*(D(2)-ksi(end))+B(end);
        B_set = [B_left; B_local_max; B_right];
        ksi_distr = [D(1);ksi_centers;D(2)];
        % figure; hold on; plot_B_margin(ksi, B, L, [D(1):0.01:D(2)]); plot(ksi_distr,B_distr,'o');
        [~,ind_max_B] =max(B_set);
        select_ksi_request = ksi_distr(ind_max_B);
    elseif strcmp(share_type,'biggest_gap')
        ksi = Loc_Acq_tuples(:,1);
        ksi_ext = [D(1);ksi;D(2)];
        ksi_diff = diff(ksi_ext);
        [~, ind] = max(ksi_diff); 
        select_ksi_request = (ksi_ext(ind)+ksi_ext(ind+1))/2;
    elseif strcmp(share_type,'urand')
        select_ksi_request =D(1)+D(2)*rand;
    end
end