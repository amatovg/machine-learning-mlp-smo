figure;
bars =  [ 8/6000  68/4000; 21/1991 33/1991];
str = {'Train Error' 'Test Error'};
bar(bars,'barWidth',0.6); 
legend('SVM','MLP');
set(gca, 'XTickLabel',str);
ylabel('Zero/One Error','FontSize',12);
title('Comparative Bar Plot','FontSize',12);