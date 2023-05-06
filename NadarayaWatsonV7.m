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
% Intrvl - ||y_est - y*||<=Intrv with prob. 1-delta

if isnan(x_meas(1))  % No data - no work!
    y_est = NaN;
    B_kt = NaN;
    h_out = NaN;
    warning('Empty data set!');
    return;
end
x_meas = x_meas(:);y_meas = y_meas(:);
x_meas = x_meas(not(isnan(x_meas)));       % work only with non-NaNs
y_meas = y_meas(not(isnan(y_meas)));       % work only with non-NaNs

 Kh = @(x,x_ms,h)(abs(x-x_ms)/h)<0.5;eps_kernel=1;
% Kh = @(x,x_ms,h)(1/sqrt(2*pi))*exp((-((x-x_ms)/h).^2)/2);eps_kernel=0.1;
%Kh = @(x,x_ms,h)(1- abs(x-x_ms)./h).*double(abs((x-x_ms)./h)<1);eps_kernel=1;
h_out = NaN;
if isnan(h_in)
    h  = 1000;
    for ii = 1:200
    K = Kh(x_arg,x_meas,h);
    K_sum = sum(K);
    if K_sum>=eps_kernel
         h_out = h;
    else 
        break;
    end
    h = h *0.9;
    end
    
end
K = Kh(x_arg,x_meas,h_out);
K_sum = sum(K);
%disp([num2str(K_sum)]);
if (K_sum>eps)&&(K_sum<=1)
        y_est = sum(y_meas.*K)/K_sum;
        B_kt = (h_out*L+2*(sigma_e/K_sum)*sqrt(log(sqrt(2)/delta)));
elseif K_sum>1
        y_est = sum(y_meas.*K)/K_sum;
        B_kt = (h_out*L+2*(sigma_e/K_sum)*sqrt(K_sum*log(sqrt(1+K_sum)/delta)));
else
    warning('Denominator equal to zero!');
    y_est = NaN;
    B_kt = NaN;
    return;
end