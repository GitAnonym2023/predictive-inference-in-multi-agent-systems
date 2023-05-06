function Acq_data = Append_Acq_dataV5(Acq_data,Shr_tuple)
   global Net_tplgy;                                      %#ok<GVMIS> 
   global M;                                              %#ok<GVMIS> 
   global L;                                              %#ok<GVMIS>
   for k = 1:M                                            % NEIGHBOR DATA COLLECTION
      Nghbrs = find(Net_tplgy(k,:));                      % Find indexes of the neighbors
      for l = Nghbrs
         if isempty(Shr_tuple{l,k}), continue; end 
         if isempty(Acq_data{k}), Acq_data{k} = Shr_tuple{l,k};break;end 
         ksi_Acq = Acq_data{k}(:,1);
         B_Acq = Acq_data{k}(:,3);
         m_Acq = Acq_data{k}(:,2);
         m_Canidate = Shr_tuple{l,k}(2);
         ksi_Candidate = Shr_tuple{l,k}(1);
         B_Canidate = Shr_tuple{l,k}(3);
         ind = abs(ksi_Acq-ksi_Candidate)<eps;
         %L_est = (B_Acq-B_Canidate)./(ksi_Acq-ksi_Candidate);
         L_est = (m_Acq-m_Canidate)./(ksi_Acq-ksi_Candidate);
         L_est(ind) = inf; 
         ind = find((abs(L_est) >= L)&(B_Acq>B_Canidate)==1);
       
           if isempty(ind)
               Acq_data{k} = [Acq_data{k};Shr_tuple{l,k}];      % Accept & save new data from the neighbors
           elseif length(ind)==1
               Acq_data{k}(ind(1),:) = Shr_tuple{l,k};           % Replace a tuple
           else
               Acq_data{k}(ind(1),:) = Shr_tuple{l,k};           % Replace and del. redundant              
               Acq_data{k}(ind(2:end),:) = [];
           end   

% Acq_data{k} = [Acq_data{k};Shr_tuple{l,k}];
       end
   end      
end
