%% Script used to generate all the data and plot it, in order to use it in the report

clear;
h1=10;eta=0.08;mu=0.3;

load('normData49.mat');
[convergence49,~,~,instance49] = TrainMLP(h1,eta,mu,1,Xtr,Ytr,Xva,Yva);
load('lastdata.mat');
tra49 = trainErrors;
val49 = validationErrors;
zoe49 = zeroOneErrors;
testErr49 = instance49.error_function(Xte,Yte);
testZoe49 = instance49.zero_one_errors(Xte,Yte);

load('normData35.mat');
[convergence35,~,~,instance35] = TrainMLP(h1,eta,mu,1,Xtr,Ytr,Xva,Yva);
load('lastdata.mat');
tra35 = trainErrors;
val35 = validationErrors;
zoe35 = zeroOneErrors;
testErr35 = instance35.error_function(Xte,Yte);
testZoe35 = instance35.zero_one_errors(Xte,Yte);

save('stage2plot1.mat','indices','tra49','val49','zoe49','tra35','val35','zoe35','convergence49','convergence35','testErr49','testZoe49','testErr35','testZoe35');
save('instance49.mat','instance49');
save('instance35.mat','instance35');

load('stage2plot1.mat');

fprintf('zoe tra 35 = %.4f zoe test 35 = %.4f\n',zoe35(convergence35+1), testZoe35);
fprintf('zoe tra 49 = %.4f zoe test 49 = %.4f\n',zoe49(convergence49+1), testZoe49);


figure(1);
subplot(1,4,1);
plot(indices,val49,'-r','LineWidth',2);
hold on;
plot(indices,tra49,'--b','LineWidth',2);
line([0 50],[testErr49 testErr49],'LineStyle','-','Color','g','LineWidth',2);
line([convergence49 convergence49],[0 1],'LineStyle','-','Color','k','LineWidth',2);
title(strcat('h1= ',num2str(h1),' eta= ',num2str(eta),' mu= ',num2str(mu)),'FontSize',12);            
xlabel('Epoch','FontSize',12);                  
ylabel('Logistic Error','FontSize',12); 
axis([0 50 0 0.15]);
grid on;                                           
legend('49 val-err','49 train-err','49 test-err');

subplot(1,4,2);
plot(indices,zoe49,'-b','LineWidth',2); 
hold on;
line([0 50],[testZoe49 testZoe49],'LineStyle','-','Color','g','LineWidth',2);
line([convergence49 convergence49],[0 1],'LineStyle','-','Color','k','LineWidth',2);
title(strcat('h1= ',num2str(h1),' eta= ',num2str(eta),' mu= ',num2str(mu)),'FontSize',12);            
xlabel('Epoch','FontSize',12);                  
ylabel('Zero/One Error','FontSize',12); 
axis([0 50 0 0.1]);
grid on;                                           
legend('49 tra-err','49 test-err');

subplot(1,4,3);
plot(indices,val35,'-r','LineWidth',2);
hold on;
plot(indices,tra35,'--b','LineWidth',2); 
line([0 50],[testErr35 testErr35],'LineStyle','-','Color','g','LineWidth',2);
line([convergence35 convergence35],[0 1],'LineStyle','-','Color','k','LineWidth',2);
title(strcat('h1= ',num2str(h1),' eta= ',num2str(eta),' mu= ',num2str(mu)),'FontSize',12);            
xlabel('Epoch','FontSize',12);                  
ylabel('Logistic Error','FontSize',12); 
axis([0 50 0 0.15]);
grid on;                                           
legend('35 val-err','3-5 tra-err','3-5 test-err');

subplot(1,4,4);
plot(indices,zoe35,'-b','LineWidth',2); 
hold on;
line([0 50],[testZoe35 testZoe35],'LineStyle','-','Color','g','LineWidth',2);
line([convergence35 convergence35],[0 1],'LineStyle','-','Color','k','LineWidth',2);
title(strcat('h1= ',num2str(h1),' eta= ',num2str(eta),' mu= ',num2str(mu)),'FontSize',12);            
xlabel('Epoch','FontSize',12);                  
ylabel('Zero/One Error','FontSize',12); 
axis([0 50 0 0.1]);
grid on;                                           
legend('35 tra-err','35 test-err');