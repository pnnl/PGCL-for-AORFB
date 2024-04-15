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
VarNameSymbol = {'\it\alpha_n','\it{a}','\sigma_m','\it{k_n}','\it{k_{m,n}}','\it{C_n}','\itD','\it{\mu_n}','\it{E_n}','\it{k_{eff}}',...
    'i_{avg}','Q','discharge','SOC','V','CaseNum'};
UD =  unique(data(:,1:10),'rows');
data_processed = [];
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
    
   if length(ind1)<=10
                continue;
   end
    count = count + 1;
    data_processed = [data_processed; ...
        [data(D(ind1),1:15) ones(length(ind1),1)*count];...
        [data(D(ind2),1:15) ones(length(ind2),1)*count];] ;
end


%% Voltage Data Visualization: Sensitive Parameter
set(0,'DefaultFigureUnits','centimeters','DefaultFigurePosition',[1 1 round(nF*10) round(nF*9)])
set(0,'DefaultAxesUnits','normalized','DefaultAxesPosition',[0.15 0.15 0.75 0.75])
SelA = 9
SelB = 3
Ind = data_processed(:,14) == 0.5 & data_processed(:,13) == 1;
A = data_processed(Ind,SelA);
B = data_processed(Ind,SelB);
C = data_processed(Ind,15);
figure,
hold on
scatter3(A,B,C,80,data_processed(Ind,15),"filled",'s','MarkerEdgeColor','k','LineWidth',1.5)
F = TriScatteredInterp(A,B,C);
xi = [min(A)*0.95:(max(A)*1.01-min(A)*0.95)/60: max(A)*1.01]
yi = [min(B)*0.95:(max(B)*1.01-min(B)*0.95)/60: max(B)*1.01]
[qx,qy] = meshgrid(xi,yi);
qz = F(qx,qy);
qz = medfilt2(qz,[8 8]);
[c,h]=contour(qx,qy,qz,10,'k','ShowText','on','LevelList',0.2:0.1:1.8)
h.LineWidth = 3;
clabel(c,h,'FontSize',15)
xlim([min(A)*0.95 max(A)*1.01])
ylim([min(B)*0.95 max(B)*1.01])
axis square
xlabel(VarNameSymbol{SelA},'interpreter','tex')
ylabel(VarNameSymbol{SelB},'interpreter','tex')
zlabel('Voltage','interpreter','tex')
colormap('cool')
view([37 30])
print -dmeta
%% Voltage Data Visualization: NonSensitive Parameter
SelA = 1
SelB = 8
Ind = data_processed(:,14) == 0.5 & data_processed(:,13) == 1;
A = data_processed(Ind,SelA);
B = data_processed(Ind,SelB);
C = data_processed(Ind,15);
figure,
hold on
scatter3(A,B,C,80,data_processed(Ind,15),"filled",'s','MarkerEdgeColor','k','LineWidth',1.5)
xlim([min(A)*0.95 max(A)*1.01])
ylim([min(B)*0.95 max(B)*1.01])
axis square
xlabel(VarNameSymbol{SelA},'interpreter','tex')
ylabel(VarNameSymbol{SelB},'interpreter','tex')
zlabel('Voltage','interpreter','tex')
colormap('cool')
view([37 30])
print -dmeta


%% Save Data
data = data_processed;
save('Voltage_Data.mat','data')
data = data_processed(data_processed(:,16)==417,:)
save('CF_Test.mat','data')

