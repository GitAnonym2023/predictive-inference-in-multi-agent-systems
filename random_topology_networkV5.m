function C = random_topology_networkV5(graph_density, seed, connectivity)
   % connectivity:'force_connectivity'
   global M; %#ok<GVMIS> 
  
   rng(seed);
   disp(['Currently used seed: ',num2str(seed)])
    C = rand(M,M);
    s=rng;s = s.Seed;
    C = C < graph_density;
    C = triu(C,1);
    %rng('shuffle')
    
    %Predefined C:
%     C = [0 1 0 0 0 0 0 0;             % C - Network Topology Matrix (only 0/1)
%          0 0 1 0 0 0 0 0;             % NOTE 1: C should Upper triangular 
%          0 0 0 1 0 0 0 0;             % NOTE 2: diag(C)=0 is required
%          0 0 0 0 1 0 0 0;
%          0 0 0 0 0 1 0 0
%          0 0 0 0 0 0 1 0
%          0 0 0 0 0 0 0 1
%          0 0 0 0 0 0 0 0];
%      M = 8;
    
    
    C = triu(C) + triu(C)';
    if strcmp(connectivity,'force_connectivity')
    not_connected = find(sum(C,2)<=1);
    
    for ii = not_connected'
        cn = randi(M-1);
        if cn==ii, cn=cn+1;end
        C(ii,cn)=1;
        C(cn,ii)=1;
        %cn2 = randi(M-1);
        %if cn2==ii || cn2==cn, cn2=cn2+1;end
        %C(ii,cn2)=1;
        %C(cn2,ii)=1;
    end
    end
    G = graph(C);     
    %figure2('units','normalized','outerposition',[0 0 1 1]);plot(G);
    figure;
    set(gcf, 'NumberTitle', 'off','Name', 'Topology');
    plot(G,"Interpreter","latex","NodeFontSize",12);set(gcf,'color','w');
%     drawnow;
    title(['Random network topology for seed: ', num2str(s)]);

end