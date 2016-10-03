%% Make a plot demonstraing the overfit effect
function GenerateOverfitPlot(convergenceEpoch)
    load('lastdata.mat');
    figure('visible','on');
    plot(indices,trainErrors,'o-r','LineWidth',2,'MarkerSize',3);
    hold on;
    plot(indices,validationErrors,'o-b','LineWidth',2,'MarkerSize',3); 
    line([convergenceEpoch convergenceEpoch],[0 1],'LineStyle','-','Color','g','LineWidth',2);
    title(strcat('4-9 dataset, h1= ',num2str(100),' eta= ',num2str(0.08),' mu= ',num2str(0.3)),'FontSize',12);            
    xlabel('Epoch','FontSize',12);                  
    ylabel('Logistic Error','FontSize',12); 
    axis([0 50 0 0.3]);
    grid on;                                           
    legend('Training error', 'Validation Error');
end