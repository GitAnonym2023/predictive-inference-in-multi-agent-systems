clear all; close all;clc; set(0,'DefaultFigureWindowStyle','docked'); %#ok<CLALL> 

global M; M = 30;               %#ok<GVMIS> Number of nodes
global D; D= [0,5];            %#ok<GVMIS> % Total inference domain (D in a paper)
global Net_tplgy;               %#ok<GVMIS> Network topology
global L; L = 1;                %#ok<GVMIS> Lipschitz constant
plot_agent = 13;                %Index of agent for plotting
N_total = 500;                % Max no. of local meas. (for every single agent)
Start_Shr = 1;                  % Start sharing when tb> Start_Shr
rng_seed = 3145.2463;   
m = @(x)cos(1.2*x).*exp(-0.3*x) + 2;% The latent nonlinear phenomenon
Net_tplgy = random_topology_networkV5(0.1, rng_seed,'force_connectivity'); % Generate network topology
new_explanatory_observationV6([], D, M,rng_seed, 'Initialize'); % Input generator initialization;    
T_hor = N_total;                % Time horizon
Sigma_e = 0.8*ones(size(1:M)).*rand(size(1:M)); % Noise disperssion
delta = 0.01;                   % All bounds with probability (1-delta)
New_meas_period = 1;            % New measurements only at every multipl. of 'New_meas_period'
Sharing_period = 1;             % Sharing is made only at every multipl. of 'Sharing_period'

delta_ksi=0.1;                  % ksi difference
ksi_predlen=10;                 % grid length

dir=zeros(M,1)+1;               % tmp direction of ksi
Dm= sort(rand(M,2)*D(2),2);     % agents local domain

%Initialization
Loc_data = nan(M,T_hor,5);      % Local explanatory and output values - k,t,(ksi,y,y_bar,B_kt) 
Shr_tuple = cell(M,M);          % Tuples selected for sharing
Req_points = zeros(M,1);        % Domain points requested by the agents from the neighbors
Acq_data = cell(M,T_hor);       % Acquired tuples (xi,y_bar,B)
ksi_mem=zeros(M,T_hor);
mu_mem=zeros(M,T_hor+1);
b_mem=zeros(M,T_hor+1);
bopt_mem=zeros(M,T_hor+1);
b_mem2=zeros(M,T_hor+1);
B_ktpred=zeros(ksi_predlen);


for t = 1:T_hor                                                  % LOCAL MEAS. AND SELECTION OF TUPLES FOR SHARING
   disp(['Time step: ',num2str(t)]);
   for k = 1:M                                                   % Select requested argument
      if mod(t,Sharing_period)==0
      Req_points(k) = select_arg2requestV6(squeeze(Loc_data(k,:,:)),Acq_data{k},'urand',0); % Select required arguments by all agents
      end
   end
   for k = 1:M                                                   % New local measurements (for all agents)
      n_local = sum(~isnan(Loc_data(k,:,1)));                    % Total number of local meas. of agent k
      if (mod(t,New_meas_period)==0)&&(n_local<N_total)          % Check whether new meas. should be taken
         n_local = n_local + 1;
        % Generating ksi   
        if t==1
            ksi_kt=(Dm(k,1)+Dm(k,2))/2;
        else
            if dir(k)==-1
                ksi_kt=ksi_mem(k,t-1)-rand(1)*delta_ksi;
            else
                ksi_kt=ksi_mem(k,t-1)+rand(1)*delta_ksi;
            end   
            if ksi_kt-delta_ksi<Dm(k,1)
                dir(k)=1;
            elseif ksi_kt+delta_ksi>Dm(k,2)
                dir(k)=-1;
            else 
                if rand(1)<0.1
                    dir(k)=-1*dir(k);
                end
            end
        end

         ksi_mem(k,t)=ksi_kt;
         y = new_output(ksi_kt, Sigma_e(k),m);                   % New local outcome data
         Loc_data(k,n_local,1:2) = [ksi_kt,y];                   % New meas. in local collection
         if t>Start_Shr
            [y_bar, B, h] = NadarayaWatsonV7(ksi_kt,Loc_data(k,:,1),Loc_data(k,:,2),NaN,L,delta,Sigma_e(k)); %NaN - for autom. h tuning
            Loc_data(k,n_local,3:5) = [y_bar,B,h];                 % New est. & B in local collection
         end
      end
      if (mod(t,Sharing_period)==0)&&(t>Start_Shr)
         Shr_tuple = share_tuple_V4(Shr_tuple, k,Loc_data,Acq_data,'',Req_points); % Select tuple for sharing
      end
      if k==plot_agent
      ksi_kpred=cat(2,linspace(ksi_kt-delta_ksi,ksi_kt,ksi_predlen/2),linspace(ksi_kt,ksi_kt+delta_ksi,ksi_predlen/2));
      [mu_pred, B_ktpred, Acq_flag]=FinalEstimateV1(ksi_kpred, Loc_data, Acq_data, B_ktpred, k, L, delta, Sigma_e);
    
      bmax=zeros([ksi_predlen,ksi_predlen]);
      bopt=zeros([10,10]);
      l1=linspace(ksi_kt-delta_ksi,ksi_kt+delta_ksi,10);
      for ii=1:10
               bopt(ii,1)=abs(m(ksi_kt)-m(ksi_kpred(ii)));
      end    
      for ii=1:ksi_predlen
          for jj=1:ksi_predlen
               bmax(ii,jj)=abs(mu_pred(ksi_predlen/2)-mu_pred(jj))+B_ktpred(k,jj);
          end
      end    

      I_row=ksi_predlen/2;
      mx=max(bmax(I_row,:));
      I_col=find(bmax(I_row,:)==mx,1,'first');
     
      [mopt,ind]=max(bopt, [], 'all');
    
        bopt_mem(k,t+1)=mopt;

       mu_mem(k,t+1)=mu_pred(I_row);
       b_mem(k,t+1)=min(mx,2*L*delta_ksi+B_ktpred(k,I_col));
       b_mem2(k,t+1)=2*L*delta_ksi+B_ktpred(k,I_col);
      end
   end
   if mod(t,Sharing_period)==0
      Acq_data = Append_Acq_dataV5(Acq_data,Shr_tuple);    

   end
      
end



k=plot_agent;
x2 = [linspace(1,N_total,(N_total)), fliplr(linspace(1,N_total,(N_total)))];   
c3=mu_mem(k,2:end)-b_mem(k,2:end);
c4=mu_mem(k,2:end)+b_mem(k,2:end);



figure(2)
         inBetween = [c3, fliplr(c4)];
         fill(x2, inBetween, [128 128 128]/255,'FaceAlpha',.4,"EdgeAlpha",0.4,'DisplayName','Prediction bounds');hold on;
         plot(linspace(0,N_total-1,N_total),m(ksi_mem(k,:)),'color',[0 0.4470 0.7410],'DisplayName','$v_{k,t}$','LineWidth', 1.5);hold on;
         plot( linspace(1,N_total,N_total),mu_mem(k,2:end),"-",'color',[0.4940 0.1840 0.5560],'DisplayName','$\hat{v}_{k,t+1|t}$','LineWidth', 1.5);
         ylim([0.5,3.5]); xlabel('$t$','interpreter','latex');
         set(gca,'fontsize',14,'TickLabelInterpreter','latex');grid on; 
         legend('Interpreter','latex');
figure(3)
         ylim([0,4]);
         plot( linspace(1,N_total,N_total),abs(b_mem(k,2:end)),'color',[128 128 128]/255,'LineWidth', 1.5,'DisplayName','Prediction bounds'); hold on;
         set(gca,'fontsize',14,'TickLabelInterpreter','latex');grid on; 
         legend('Interpreter','latex');

