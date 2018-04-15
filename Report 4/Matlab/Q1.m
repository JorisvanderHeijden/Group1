% Neural network
clear variables;clc;close all; syms z
w=[2 1 2 4 1]
x=2;y=3;
sigma=matlabFunction(1/(1+exp(-z)));
sigma_der=matlabFunction(exp(-z)/(exp(-z)+1)^2);


for i=1
    z1=w(1)*x;
    n1=sigma(z1);
    z2=n1*w(2);n2=sigma(z2);
    z3=n1*w(3);n3=sigma(z3);
    z4=n2*w(4)+n3*w(5); n4=sigma(z4);

    Error(i)=1/2*(n4-y)^2;
    beta=(n4-y)*sigma_der(z4);
    dl_dw(5)=beta*n3;
    dl_dw(4)=beta*n2;
    dl_dw(3)=beta*w(5)*sigma_der(z3)*n1;
    dl_dw(2)=beta*w(4)*sigma_der(z2)*n1;

    dl_dw1_1=beta*w(4)*sigma_der(z2)*w(2)*sigma_der(z1)*x;
    dl_dw1_2=beta*w(4)*sigma_der(z3)*w(3)*sigma_der(z1)*x;

    dl_dw(1)=(dl_dw1_1+dl_dw1_2);

    alfa=0.1;
    wnew=w-alfa*dl_dw

    w=wnew;

end

plot(Error)
