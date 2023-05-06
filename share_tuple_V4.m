function Shr_tuple = share_tuple_V4(Shr_tuple,k, Loc_tuples, Acq_tuples, sharing_type, Req_points)
    % TODO: sharing_type: 'share_urand', 'share_best', 'share_worst',  'share_requested'
    global Net_tplgy; %#ok<GVMIS> 
    N_k = find(Net_tplgy(k,:));                         % A set of agent k neighbors
    for nghb = N_k                                      % Go through all the neighbors and select tuples for sharing
       Local_k = squeeze(Loc_tuples(k,:,[1,3,4,5]));      % Do not share raw output meas. (2)
       Acqrd_k = squeeze(Acq_tuples{k});
       L_ind = not(isnan(sum(Local_k,2)));              % Find all well defined acquired tuples
       A_ind = not(isnan(sum(Acqrd_k,2)));
       Candidates_to_share_by_k = [Local_k(L_ind,:);Acqrd_k(A_ind,:)];       
       Tot_num = size(Candidates_to_share_by_k,1);
       if Tot_num == 0, break; end
       [~, share_ind] = min(abs(Candidates_to_share_by_k(:,1)-Req_points(nghb)));
       ksi = Candidates_to_share_by_k(share_ind,1);
       y_bar = Candidates_to_share_by_k(share_ind,2);
       B = Candidates_to_share_by_k(share_ind,3);
       h = Candidates_to_share_by_k(share_ind,4);
       Shr_tuple{k,nghb} = [ksi,y_bar,B,h];
    end
end