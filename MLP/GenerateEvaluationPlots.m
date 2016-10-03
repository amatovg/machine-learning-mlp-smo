% The execution of the script MLPEvaluationStage1.m is required before
% executing this script. In this script the only thing which happens is the
% generation of the plots for the report for the part of MLP Evaluation.

load('h1_35.mat')
load('h1_49.mat')

f = figure(1);
set(f, 'Position', [400 400 900 300])
subplot(1,2,1);
plot(h1,h1_35_values,'o-b','LineWidth',2,'MarkerSize',3)
hold on;
plot(h1,h1_35_tr_values,'--b','LineWidth',2)
plot(h1,h1_49_values,'o-r','LineWidth',2,'MarkerSize',3)
plot(h1,h1_49_tr_values,'--r','LineWidth',2)
hold off;
title('eta = 0.08 mu = 0.2','FontSize',12);
legend('3-5 validation','3-5 training','4-9 validation','4-9 training');
xlabel('h1','FontSize',12);                  
ylabel('Logistic Error','FontSize',12); 
axis([0 100 0 0.1]);
grid on;

subplot(1,2,2);
plot(h1,h1_35_epochs,'o-r','LineWidth',2,'MarkerSize',3)
hold on;
plot(h1,h1_49_epochs,'o-b','LineWidth',2,'MarkerSize',3)
hold off;
title('eta = 0.08 mu = 0.2','FontSize',12);
legend('3-5 dataset','4-9 dataset');           
xlabel('h1','FontSize',12);                  
ylabel('Convergence Epoch','FontSize',12); 
axis([0 100 0 14]);
grid on; 

load('eta_35.mat')
load('eta_49.mat')

g = figure(2);
set(g, 'Position', [400 400 900 300])
subplot(1,2,1);
plot(eta,eta_35_values,'o-b','LineWidth',2,'MarkerSize',3)
hold on;
plot(eta,eta_35_tr_values,'--b','LineWidth',2)
plot(eta,eta_49_values,'o-r','LineWidth',2,'MarkerSize',3)
plot(eta,eta_49_tr_values,'--r','LineWidth',2)
hold off;
title('h1 = 10 mu = 0.2','FontSize',12);
legend('3-5 validation','3-5 training','4-9 validation','4-9 training');
xlabel('eta','FontSize',12);                  
ylabel('Logistic Error','FontSize',12); 
axis([0 0.5 0 0.2]);
grid on;

subplot(1,2,2);
plot(eta,eta_35_epochs,'o-b','LineWidth',2,'MarkerSize',3)
hold on;
plot(eta,eta_49_epochs,'o-r','LineWidth',2,'MarkerSize',3)
hold off;
title('h1 = 10 mu = 0.2','FontSize',12);
legend('3-5 dataset','4-9 dataset');           
xlabel('eta','FontSize',12);                  
ylabel('Convergence Epoch','FontSize',12); 
axis([0 0.5 0 20]);
grid on; 


load('mu_35.mat')
load('mu_49.mat')

h = figure(3);
set(h, 'Position', [400 400 900 300])
subplot(1,2,1);
plot(mu,mu_35_values,'o-b','LineWidth',2,'MarkerSize',3)
hold on;
plot(mu,mu_35_tr_values,'--b','LineWidth',2,'MarkerSize',3)
plot(mu,mu_49_values,'o-r','LineWidth',2,'MarkerSize',3)
plot(mu,mu_49_tr_values,'--r','LineWidth',2,'MarkerSize',3)
hold off;
title('h1 = 10 eta = 0.08','FontSize',12);
legend('3-5 validation','3-5 training','4-9 validation','4-9 training');
xlabel('mu','FontSize',12);                  
ylabel('Logistic Error','FontSize',12); 
axis([0 1 0 0.1]);
grid on;

subplot(1,2,2);
plot(mu,mu_35_epochs,'o-b','LineWidth',2,'MarkerSize',3)
hold on;
plot(mu,mu_49_epochs,'o-r','LineWidth',2,'MarkerSize',3)
hold off;
title('h1 = 10 eta = 0.08','FontSize',12);
legend('3-5 dataset','4-9 dataset');           
xlabel('mu','FontSize',12);                  
ylabel('Convergence Epoch','FontSize',12); 
axis([0 1 0 10]);
grid on; 