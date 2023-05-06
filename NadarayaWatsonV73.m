function [y_est, B_kt, h_out] = NadarayaWatsonV7 (x_arg, x_meas, y_meas, h_in, L, delta, sigma_e)
    
    % INPUT:
    % x_arg - estimation point
    % x_meas - set of explanatory measurements
    % y_meas - set of output measurements
    % h_in - bandwidth, if NaN then selftuning
    % L - nonlinearity Lipshitz constant
    % delta - probability guaranties for estimate margins (see outputs)
    % sigma_n - noise signal disperssion
    % OUTPUT:
    % y_est - output estimate
    % B_kt - ||y_est - y*||<=Intrv with prob. 1-delta
    
    kNN_Threshold = 1;
    if isnan(x_meas(1))  % No data - no work!
        y_est = NaN;
        B_kt = NaN;
        h_out = NaN;
        warning('Empty data set!');
        return;
    end
    if sum(~isnan(x_meas))<kNN_Threshold
       y_est = NaN;
       B_kt = NaN;
       h_out = NaN;
       return
    end
    x_meas = x_meas(:);y_meas = y_meas(:);
    x_meas = x_meas(not(isnan(x_meas)));       % work only with non-NaNs
    y_meas = y_meas(not(isnan(y_meas)));       % work only with non-NaNs

    Kh = @(x,x_ms,h)(abs(x-x_ms)/h)<0.5;
    % Kh = @(x,x_ms,h)(1/sqrt(2*pi))*exp((-((x-x_ms)/h).^2)/2);
   % Kh = @(x,x_ms,h)(1- abs(x-x_ms)./h).*double(abs((x-x_ms)./h)<1);
    h = 1000;
    B_kt = inf;
    y_est = 0;
    if isnan(h_in)
        for ii = 1:200              
            K = Kh(x_arg,x_meas,h);
            [B_kt_new, y_est_new] = Eval_Bkt_ykt(K, y_meas, sigma_e, delta, h, L);
            if B_kt_new <= B_kt
                B_kt = B_kt_new;
                y_est = y_est_new;
                h_out = h;
                h = h*0.9;
                if (sum(K>0)<kNN_Threshold)
                    break;
                end
            else
                break
            end
        end
%     if (sum(K>0)<kNN_Threshold)
%         y_est = NaN;
%         B_kt = NaN;
%         h_out = NaN;
%     end
    else
        K = Kh(x_arg,x_meas,h_in);
        [B_kt, y_est] = Eval_Bkt_ykt(K, y_meas, sigma_e, delta, h_in, L);
        h_out = h_in;
    end
end

function [B_kt, y_est] = Eval_Bkt_ykt(K, y_meas, sigma_e, delta, h_in,L)
    K_sum = sum(K);
    if (K_sum>eps)&&(K_sum<=1)
        y_est = sum(y_meas.*K)/K_sum;
        B_kt = (h_in*L+2*(sigma_e/K_sum)*sqrt(log(sqrt(2)/delta)));
    elseif K_sum>1
        y_est = sum(y_meas.*K)/K_sum;
        B_kt = (h_in*L+2*(sigma_e/K_sum)*sqrt(K_sum*log(sqrt(1+K_sum)/delta)));
    else
        y_est = NaN;
        B_kt = NaN;
    end
end