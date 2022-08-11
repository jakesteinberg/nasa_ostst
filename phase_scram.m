% first attempt at phase scrambling 

addpath '/Users/jakesteinberg/Documents/matlab'/glider_toolkit_regressions_jun19/
%%
tg = load('/Users/jakesteinberg/Documents/NASA_OSTST/tide_gauge/GOM_gauges_long.mat');
data = tg.height_grid/1000; % - nanmean(tg.height_grid/1000,2);
time = tg.t_grid;
%%
id_tags_short = {'KEY WEST','NAPLES','FORT MYERS','ST. PETERSBURG','CLEARWATER BEACH',...
    'CEDAR KEY II', 'APALACHICOLA', 'PANAMA CITY, ST.ANDREWS BAY, FL', 'PENSACOLA', ...
    'DAUPHIN ISLAND', 'MOBILE', 'BAY WAVELAND YACHT CLUB' ,'SHELL BEACH, LOUISIANA', ...
    'GRAND ISLE', 'SABINE PASS', 'GALVESTON', 'FREEPORT', 'CORPUS CHRISTI, GULF MEXICO, TX', 'PORT ISABEL'};
%%
% indi = 6210:9861; % (2010-2019)
indi = 6224:9845; % (15-01-2010, 15-12-2019)
test = data(1:12,indi);
test2 = test;
test_time = time(indi);
for i = 1:size(test,1)
    test2(i,:) = nanseginterp(test_time,test(i,:));
end
test3 = test2 - nanmean(test2,2);
sum(sum(isnan(test2)))
%%
u = test3(1,:)';
[n,m]=size(u);
n2=fix(n/2);
if 2*n2 ~= n
    n2=n2+1;
end
% Transform u
U=fft(u);
% Find radius and random phase for first n2 data
R=abs(U(1:n2,:));
theta=rand(n2,m)*2*pi; % uniform distribution of phase
P=exp(1i*theta); % this is faster than cos(theta) + i*sin(theta)
U1top=R.*P; % top of FT
U2bot=conj(flipud(U1top)); % bottom of FT is complex conj of top
if 2*n2 == n
    U1=[U1top;U(n2+1,:);U2bot(1:n2-1,:)];
else
    % when n is odd we just butt the 2 arrays together
    U1=[U1top;U2bot(1:n2-1,:)];
end
% Set the first component (ie mean) of both arrays to be same
U1(1,:)=U(1,:);
% Transform back and take real part (imag is zero, but no is complex)
u1=real(ifft(U1));
%%
u = test2'; 
trend0 = nan*ones(1,size(u,2));
for i = 1:size(u,2)
    fit = polyfit(test_time,u(:,i),1);
    trend0(i) = fit(1)*365;
end
trend = nan*ones(100,size(u,2));
for i = 1:1500
    u1=pharand(u);
    for j = 1:size(u1,2)
        fit = polyfit(test_time,u1(:,j),1); 
        trend(i,j) = fit(1)*365;
    end
end
%%
figure('units','inches','position',[1 1 14 9]);
for i = 1:size(u,2)
    subplot(3,4,i); hold on;
    histogram(trend(:,i),50,'Normalization','probability')
    plot([trend0(i),trend0(i)],[0,125],'--r')
    plot([mean(trend(:,i)),mean(trend(:,i))],[0,125],'--k')
    xlim([-0.018,0.018])
    ylim([0,0.1])
    title(id_tags_short(i))
    if i > 8
        xlabel('2010-2019 decadal trend [m/yr]')
    end
    if i == 9
        ylabel('probability density')
    end
end
%%
p_val = nan*ones(1,size(u,2));
for i = 1:size(u,2)
    [t_z,t_mu,t_sig] = zscore([trend(:,i);trend0(i)]);
    p_val(i) = 1 - normcdf(t_z(end)); % this a right tailed dist (alt. hypoth is >), two tailed = (1-normcdf(t_z(end))) * 2
end


% https://www.statology.org/matlab-z-score-to-p-value/

