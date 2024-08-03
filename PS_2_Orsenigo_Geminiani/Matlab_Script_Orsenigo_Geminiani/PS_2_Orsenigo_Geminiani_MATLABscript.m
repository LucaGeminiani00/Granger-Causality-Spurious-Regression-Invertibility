%%%%%%%%%%%%%%%%%%%%%% Exercise 1 %%%%%%%%%%%%%%%%%%%%%%

%regression and unit root test with an intercept
t_test = zeros(100000,1);
alpha=4;
for i = 1:100000 
    y = zeros(250,1);
    y(1)= alpha;
    epsilon = sqrt(0.2) * randn(250, 1); 
    for t = 2:250
        y(t) = alpha+ y(t-1) + epsilon(t);
    end 
    vx = [ones(249,1) y(1:249)];
    vy = y(2:250);
    [beta,res,cov_b] = OLS(vy,vx);
    t_test(i) = (beta(2)-1)/sqrt(cov_b(2,2));
end 

critical = tinv(1-(0.05/2),250-2);
sum(abs(t_test)>critical)/100000 
figure;histogram(t_test)
mean(t_test)

%regression and unit root test with an intercept and a time trend

t_test = zeros(100000,1);
alpha=4;
delta=1;
for i = 1:100000 
    y = zeros(250,1);
    y(1)= alpha;
    epsilon = sqrt(0.2) * randn(250, 1); 
    for t = 2:250
    y(t) = alpha+ y(t-1) + delta*(t-1) +epsilon(t); 
    %delta (t-1) perchè l'intercept t=0 è in y(1)%
    end 
    vx = [ones(249,1) y(1:249) (1:249)'];
    vy = y(2:250);
    [beta,res,cov_b] = OLS(vy,vx);
    t_test(i) = (beta(2)-1)/sqrt(cov_b(2,2));
end 

critical = tinv(1-(0.05/2),250-3);
sum(abs(t_test)>critical)/100000 
figure;histogram(t_test)

%%%%%%%%%%%%%%%%%%%%%% Exercise 2 %%%%%%%%%%%%%%%%%%%%%%
clear

%1st case
T = 250;          
rho1 = 0.7;   
rho2 = 0.2; 
MC = 5000; 
rho = 1; 
Y = zeros(T,1);
for t = 1:MC 
    e = randn(T,1);
    e(1)=0;
    w = randn(T,1); 
    w(1)=0;
    y = filter(1,[1 -rho1],e);
    Z = filter(1,[1 -rho2],w); 
    X = [ones(T, 1), Z]; 
    beta_hat1(:,t) = X \ y;
    iXX = inv(X'*X);
    sig_hat(t) = cov(y - X*beta_hat1(:,t));               
    s_beta = iXX(2,2)*sig_hat(t);                      
    t_ratio1(t) = (beta_hat1(2,t))/sqrt(s_beta);
    residuals = y - X * beta_hat1(:, t);
    y_bar = mean(y);
    tss = sum((y - y_bar).^2);
    rss = sum(residuals.^2);
    r_squared1(t) = 1 - rss / tss;
end 

%2n case

T = 250;          
rho1 = 0.7;   
rho2 = 1; 
MC = 5000; 
alpha = 0.6;
rho = 1; 
Y = zeros(T,1);
for t = 1:MC
    e = randn(T,1);
    e(1)=0;
    w = randn(T,1); 
    w(1)=0;
    y = filter(1,[1 -rho1],e);
    Z = filter(1,[1 -rho2],w);
    X = [ones(T, 1), Z];
    beta_hat2(:,t) = X \ y;
    iXX = inv(X'*X);
    sig_hat(t) = cov(y - X*beta_hat2(:,t));               
    s_beta = iXX(2,2)*sig_hat(t);                      
    t_ratio2(t) = (beta_hat2(2,t))/sqrt(s_beta);
    residuals = y - X * beta_hat2(:, t);
    y_bar = mean(y);
    tss = sum((y - y_bar).^2);
    rss = sum(residuals.^2);
    r_squared2(t) = 1 - rss / tss;
end 

%3rd case

T = 250;          
rho = 1;      
MC = 5000; 
alpha = 0.6;
rho = 1; 
Y = zeros(T,1);
for t = 1:MC
    e = randn(T,1);
    e(1)=0;
    w = randn(T,1);
    w(1)=0;
    y = filter(1,[1 -rho],e);
    Z = filter(1,[1 -rho],w);
    X = [ones(T, 1), Z]; 
    beta_hat3(:,t) = X \ y;
    iXX = inv(X'*X);
    sig_hat(t) = cov(y - X*beta_hat3(:,t));               
    s_beta = iXX(2,2)*sig_hat(t);                      
    t_ratio3(t) = (beta_hat3(2,t))/sqrt(s_beta);
    residuals = y - X * beta_hat3(:, t);
    y_bar = mean(y);
    tss = sum((y - y_bar).^2);
    rss = sum(residuals.^2);
    r_squared3(t) = 1 - rss / tss;
end 

%4th case 

T = 250;          
rho = 1; 
MC = 5000; 
rho = 1; 
Y = zeros(T,1);
for t = 1:MC
    w = randn(T,1);
    w(1)=0;
    y = filter(1,[1 -rho],w);
    Z = filter(1,[1 -rho],w);
    X = [ones(T, 1), Z];
    beta_hat4(:,t) = X \ y;
    iXX = inv(X'*X);
    sig_hat(t) = cov(y - X*beta_hat4(:,t));               
    s_beta = iXX(2,2)*sig_hat(t);                      
    t_ratio4(t) = (beta_hat4(2,t))/sqrt(s_beta);
    residuals = y - X * beta_hat4(:, t);
    y_bar = mean(y);
    tss = sum((y - y_bar).^2);
    rss = sum(residuals.^2);
    r_squared4(t) = 1 - rss / tss;
end 

%results 

figure
[f,xi] = ksdensity(beta_hat1(2,:));
plot(xi,f,'LineWidth', 2);
hold on
[f,xi] = ksdensity(beta_hat2(2,:));
plot(xi,f,'LineWidth', 2);
[f,xi] = ksdensity(beta_hat3(2,:));
plot(xi,f,'LineWidth', 2);
hold off

critical = tinv(1-(0.05/2),250-2);
sum(abs(t_ratio1)>critical)/5000
sum(abs(t_ratio2)>critical)/5000
sum(abs(t_ratio3)>critical)/5000
sum(abs(t_ratio4)>critical)/5000

figure
[f,xi] = ksdensity(r_squared1);
plot(xi,f,'LineWidth', 2);
hold on
[f,xi] = ksdensity(r_squared2);
plot(xi,f,'LineWidth', 2);
[f,xi] = ksdensity(r_squared3);
plot(xi,f,'LineWidth', 2);
hold off

mean(r_squared1);
mean(r_squared2);
mean(r_squared3);

mean(beta_hat4(2,:))
mean(r_squared4)

%%%%%%%%%%%%%%%%%%%%%% Exercise 3 %%%%%%%%%%%%%%%%%%%%%%
clear 

data1= xlsread("Romer_Romer.xlsx");
data=data1(:,2:end); %Removing column with dates

depvar = data(5:end,1);
lag1 = data(4:end-1,1:end);
lag2 = data(3:end-2,1:end); 
lag3 = data(2:end-3,1:end);
lag4 = data(1:end-4,1:end);
X = [lag1 , lag2 , lag3 , lag4]; 
b = inv(X'*X)*X'*depvar; 
res = depvar-X*b;
cov_b = inv(X'*X)*cov(res);
t_test=zeros(16,1);
for t=1:16
t_test(t)=b(t)/sqrt(cov_b(t,t));
end
critical = tinv(1-(0.05/32),108-16);
abs(t_test)>critical

depvar = data(5:end,2);
lag1 = data(4:end-1,1:end);
lag2 = data(3:end-2,1:end); 
lag3 = data(2:end-3,1:end);
lag4 = data(1:end-4,1:end);
X = [lag1 , lag2 , lag3 , lag4]; 
b = inv(X'*X)*X'*depvar; 
res = depvar-X*b;
cov_b = inv(X'*X)*cov(res);
for t=1:16
t_test(t)=b(t)/sqrt(cov_b(t,t));
end
critical = tinv(1-(0.05/32),108-16);
abs(t_test)>critical

depvar = data(5:end,3);
lag1 = data(4:end-1,1:end);
lag2 = data(3:end-2,1:end); 
lag3 = data(2:end-3,1:end);
lag4 = data(1:end-4,1:end);
X = [lag1 , lag2 , lag3 , lag4]; 
b = inv(X'*X)*X'*depvar; 
res = depvar-X*b;
cov_b = inv(X'*X)*cov(res);
for t=1:16
t_test(t)=b(t)/sqrt(cov_b(t,t));
end
critical = tinv(1-(0.05/32),108-16);
abs(t_test)>critical

%%%%%%%%%%%%%%%%%%%%%% Exercise 4 %%%%%%%%%%%%%%%%%%%%%%
clear 

T = 600; 
vary = 0.8; 
beta = 0.6; 
N = 500;
Imp0 = cell([1 N]);
Imp1 = cell([1 N]);
Imp2 = cell([1 N]);
Imp3 = cell([1 N]);
Imp4 = cell([1 N]);

for i = 1:N
   eta = randn(T,1); 
   epsilon = randn(T,1)*sqrt(vary) ;
   X = zeros(T,1);
   Y = zeros(T,1);
   for t = 3:T
       x(t) = eta(t) + epsilon(t-2); 
       y(t) = (beta/(1-beta))*eta(t) + (beta^2/(1-beta))*epsilon(t)+beta*epsilon(t-1); 
   end 
   G = x(101:600); 
   Y = y(101:600);
   data = [G ; Y]'; 

   depvar = data(5:end,1);  %Regression for X 
   lag1 = data(4:end-1,:); 
   lag2 = data(3:end-2,:); 
   lag3 = data(2:end-3,:);
   lag4 = data(1:end-4,:); 
   X = [lag1 , lag2 , lag3 , lag4];  
   b = inv(X'*X)*X'*depvar; 
   res1 = depvar-X*b;
   cov_b = inv(X'*X)*cov(res1);
    
   depvar = data(5:end,2);  %Regression for y 
   lag1 = data(4:end-1,:); 
   lag2 = data(3:end-2,:); 
   lag3 = data(2:end-3,:);
   lag4 = data(1:end-4,:); 
   X = [data(5:end,1), lag1 , lag2 , lag3 , lag4];  
   b2 = inv(X'*X)*X'*depvar;
   res2 = depvar-X*b2;
   cov_b = inv(X'*X)*cov(res2);

   V = [res1 res2];
   I = eye(2);
   C = [1 0; b2(1) 1]; 
   A1 =C*[b(1) b(2); b2(2) b2(3)]; 
   A2 = C*[b(3) b(4);b2(4) b2(5)]; 
   A3 = C*[b(5) b(6);b2(6) b2(7)];
   A4 = C*[b(7) b(8);b2(8) b2(9)];
   O = zeros(2,2);
   A = [A1 A2 A3 A4; I O O O; O I O O; O O I O];
   D = [C; O; O; O];
   
   a_1 = A*D;
   a_2 = (A^2)*D;
   a_3 = (A^3)*D;
   a_4 = (A^4)*D;
   
   Imp0{i} = [D(1,:); D(2,:); D(3,:); D(4,:)];
   Imp1{i} = [a_1(1,:); a_1(2,:); a_1(3,:); a_1(4,:)];
   Imp2{i}= [a_2(1,:); a_2(2,:); a_2(3,:); a_2(4,:)];
   Imp3{i} = [a_3(1,:); a_3(2,:); a_3(3,:); a_3(4,:)];
   Imp4{i} = [a_4(1,:); a_4(2,:); a_4(3,:); a_4(4,:)];

end

%Compute means of the impulse responses. 

ImpAvg0 = zeros(4,2);
ImpAvg1 = zeros(4,2);
ImpAvg2 = zeros(4,2);
ImpAvg3 = zeros(4,2);
ImpAvg4 = zeros(4,2);

for j = 1:N
     ImpAvg0 = ImpAvg0 + Imp0{j};
     ImpAvg1 = ImpAvg1 + Imp1{j};
     ImpAvg2 = ImpAvg2 + Imp2{j};
     ImpAvg3 = ImpAvg3 + Imp3{j};
     ImpAvg4 = ImpAvg4 + Imp4{j};
end

MeanImp0 = ImpAvg0/N ; 
MeanImp1 = ImpAvg1/N ;
MeanImp2 = ImpAvg2/N ;
MeanImp3 = ImpAvg3/N ;
MeanImp4 = ImpAvg4/N ;

%PLOT THE RESULTS: We compare the true impulse response functions to the

%Impulse Response of X to a unitary change in eta 
subplot(2,3,1);
time = [0:4];
X1ETA = [MeanImp0(1,1) MeanImp1(1,1) MeanImp2(1,1) MeanImp3(1,1) MeanImp4(1,1)];
plot(time, X1ETA, 'black');
hold on 
TXETA = [1 0 0 0 0]; % True impulse response
plot(time, TXETA, 'red');
xlabel('time')
ylabel('x')
hold off

%IRF of x to a 1 unit change in epsilon 
subplot(2,3,2);
X1EPS = [MeanImp0(1,2) MeanImp1(1,2) MeanImp2(1,2) MeanImp3(1,2) MeanImp4(1,2)];
plot(time, X1EPS, 'black');
hold on 
TXEPS = [0 0 1 0 0]; % True impulse response
plot(time, TXEPS, 'red');
xlabel('time')
ylabel('x')
hold off

%IRF of x to a 1 unit change in both eta and epsilon
subplot(2,3,3);
X1EPSETA = [MeanImp0(1,1)+MeanImp0(1,2), MeanImp1(1,1)+ MeanImp1(1,2), MeanImp2(1,1)+ MeanImp2(1,2), MeanImp3(1,1)+MeanImp3(1,2), MeanImp4(1,1)+MeanImp4(1,2)];
plot(time, X1EPSETA, 'black')
hold on 
TXEPSETA = [1 0 1 0 0]; % True impulse response
plot(time, TXEPSETA, 'red');
xlabel('time')
ylabel('x')
hold off

%IRF Of y to a unit change in eta 
subplot(2,3,4);
Y1ETA = [MeanImp0(2,1) MeanImp1(2,1) MeanImp2(2,1) MeanImp3(2,1) MeanImp4(2,1)];
plot(time, Y1ETA, 'black');
hold on 
TYETA = [beta/(1-beta) 0 0 0 0]; % True impulse response
plot(time, TYETA, 'red');
xlabel('time')
ylabel('y')
hold off

%IRF of y to a 1 unit change in epsilon 
subplot(2,3,5);
Y1EPS = [MeanImp0(2,2) MeanImp1(2,2) MeanImp2(2,2) MeanImp3(2,2) MeanImp4(2,2)];
plot(time, Y1EPS, 'black');
hold on 
TYEPS = [beta^2/(1-beta) beta 0 0 0]; % True impulse response
plot(time, TYEPS, 'red');
xlabel('time')
ylabel('y')
hold off

%IRF of y to a 1 unit change in both eta and epsilon 
subplot(2,3,6);
Y1EPSETA = [MeanImp0(2,1)+MeanImp0(2,2), MeanImp1(2,1)+ MeanImp1(2,2), MeanImp2(2,1)+ MeanImp2(2,2), MeanImp3(2,1)+MeanImp3(2,2), MeanImp4(2,1)+MeanImp4(2,2)];
plot(time, Y1EPSETA, 'black')
hold on 
TYEPSETA = [beta^2/(1-beta)+beta/(1-beta), beta, 0  0 0]; % True impulse response
plot(time, TYEPSETA, 'red');
xlabel('time')
ylabel('y')
hold off

%Percentile Computation 
c1 = zeros(1,N);
for j = 1:N 
    p = Imp1{j}; 
    c1(j) = p(1,1);  
end 
P1 = prctile(c1,5);  
P2 = prctile(c1,95);  