clc;
clear;
close all;
%Set the plot parameters**********************************
nF=2;
set(0,'DefaultFigureUnits','centimeters','DefaultFigurePosition',[1 1 round(nF*10) round(nF*9)])
set(0,'DefaultFigureColor',[1 1 1])
set(0,'DefaultAxesUnits','normalized','DefaultAxesPosition',[0.2 0.2 0.75 0.75])
%set(0,'DefaultAxesXTickMode','manual','DefaultAxesYTickMode','manual')
set(0,'DefaultAxesTickLength',[0.02 0.02])
set(0,'DefaultAxesXMinorTick','on','DefaultAxesYMinorTick','on')
set(0,'DefaultAxesLineWidth',ceil(nF*1),'DefaultAxesFontName','Times',...
    'DefaultAxesFontSize',ceil(nF*12),'DefaultAxesBox','on')
set(0,'DefaultLineLineWidth',ceil(nF*1),'DefaultLineMarkerSize',ceil(nF*6))
set(0,'DefaulttextFontName','Times','DefaulttextFontSize',nF*8)
%Set the plot parameters**********************************

addpath(genpath('./Matlab_Functions'))

%% Read ID 780 cm2 cell Data
data = xlsread('ID 780 Cell Dataset.xlsx','A2:O38913');
data(isnan(data(:,15)),:) = []; % delete nan data
data(data(:,15)>2,:) = [];
data(data(:,15)<0,:) = [];
VarName = ["alpha_n","a_n","sigma_m","k_n","km_n","C_n","D","mu_n","E_n","keff",...  % 1-10
    "i_avg","Q","discharge","SOC","Voltage" ]; %11-15



%% Data Processing
% Calcualted Energy Efficiency (EE)
VarNameSymbol = {'\it\alpha_n','\it{a}','\sigma_m','\it{k_n}','\it{k_{m,n}}','\it{C_n}','\itD','\it{\mu_n}','\it{E_n}','\it{k_{eff}}',...
    'i_{avg}','Q','discharge','SOC','V','CaseNum'};
UD =  unique(data(:,1:10),'rows');
count = 0;
for k = 1:size(UD,1)
    D =find(ismember(data(:,1:10),UD(k,:),'rows'));
    UD_size(k) = length(D);
    ind1 = find(data(D,13) == 1); % charge
    ind2 = find(data(D,13) == -1); % discharge
    SOC1 = data(D(ind1),14);
    SOC2 = data(D(ind2),14);
    [~, s1, s2] = intersect(SOC1, SOC2,'rows');
    ind1 = ind1(s1);
    ind2 = ind2(s2);
    EE(k) = sum(data(D(ind2),15))/sum(data(D(ind1),15));
    count = count + 1;
end

UD(isnan(EE),:) = [];
EE(isnan(EE)) = [];


%% EE Data Visualization - Sensitive Parameter
SeqUD = [1 8 9 2 10 3 4 5 6 7];
X = UD;
Y = EE';
set(0,'DefaultFigureUnits','centimeters','DefaultFigurePosition',[1 1 round(nF*10) round(nF*9)])
set(0,'DefaultAxesUnits','normalized','DefaultAxesPosition',[0.15 0.15 0.75 0.75])
SelA = 9
SelB = 3
A = X(:,SelA);
B = X(:,SelB);

figure,
hold on
scatter3(X(:,SelA),X(:,SelB),Y,80,Y,"filled",'s','MarkerEdgeColor','k','LineWidth',1.5)
F = TriScatteredInterp(A,B,Y);
xi = [min(A)*0.95:(max(A)*1.01-min(A)*0.95)/60: max(A)*1.01]
yi = [min(B)*0.95:(max(B)*1.01-min(B)*0.95)/60: max(B)*1.01]
[qx,qy] = meshgrid(xi,yi);
qz = F(qx,qy);
qz = medfilt2(qz,[8 8]);
[~,c]=contour(qx,qy,qz,10,'k','ShowText','on','LevelList',0.5:0.05:1)
c.LineWidth = 3;
xlim([min(A)*0.95 max(A)*1.01])
ylim([min(B)*0.95 max(B)*1.01])
axis square
xlabel(VarNameSymbol{SelA},'interpreter','tex')
ylabel(VarNameSymbol{SelB},'interpreter','tex')
zlabel('Energy Efficiency','interpreter','tex')
colormap('autumn')
view([37 30])

%% EE Data Visualization - Non Sensitive Parameter
SelA = 8
SelB = 10
A = X(:,SelA);
B = X(:,SelB);
figure,
scatter3(X(:,SelA),X(:,SelB),Y,80,Y,"filled",'s','MarkerEdgeColor','k','LineWidth',1.5)
xlim([min(A)*0.95 max(A)*1.01])
ylim([min(B)*0.95 max(B)*1.01])
axis square
xlabel(VarNameSymbol{SelA},'interpreter','tex')
ylabel(VarNameSymbol{SelB},'interpreter','tex')
zlabel('Energy Efficiency','interpreter','tex')
colormap('autumn')
caxis([0.5 1]);
view([37 30])

%% Save Data
data = [UD EE'];
save('EE_data.mat','data')
