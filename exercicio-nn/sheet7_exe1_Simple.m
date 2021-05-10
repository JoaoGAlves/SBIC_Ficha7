clf;

syms x;

xa = zeros(1,1000);
xb = xa; %inicializar xa e xb
yd = xb;
somatorio= yd;
somatorio_t = yd;
y = yd;
xa_t=xa;
xb_t=xb;
y_t = yd;
vetor_erro = y_t;

erro_desejado = ones(1,200).*0.15

cnt = 1;

w(1) = -0.5 + 0.5*rand(1,1);
w(2) = -0.5 + 0.5*rand(1,1);
w(3) = -0.5 + 0.5*rand(1,1);
alpha = 0.05;
theta = 1;

%%%%%%%%%%%%%%%%ler txt%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sizeA = [3 1000];
sizeB = [2 1000];
fileID = fopen('testInput10A.txt', 'r');

[A,count] = fscanf(fileID, '%f,%f,%d', sizeA); %lê tudo que é para treino

B=[A(1,length(A)); A(2,length(A))];
[B1, count] = fscanf(fileID, '%f,%f', sizeB);
B=[B(1,1), B1(1,:);B(2,1), B1(2,:)]; %lê tudo que é para testar
fclose(fileID);
A=A';
B=B';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%retirar dados%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:size(A, 1) %retira valores depois de encontrar 0
   if(A(i,3) == 0)
       s = i;
       break;   
   else
       xa(i) = A(i,1);
       xb(i) = A(i,2);
       yd(i) = A(i,3); %output desejado
   end
end

for i=1:1:size(B, 1) %retira valores depois de encontrar 0       
    xa_t(i) = B(i,1);
    xb_t(i) = B(i,2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xa = xa(1:1:s-1); %truncar 
xb = xb(1:1:s-1);
xa_t = xa_t(1:1:length(B));
xb_t = xb_t(1:1:length(B));
%%%%%%%%%%%%%%%%normalizar dados de input%%%%%%%%%%%%%%%%%%%
maximo = max([xa,xb]);
minimo = min([xa,xb]);
xa = (xa - minimo)./(maximo-minimo);
xb = (xb - minimo)./(maximo-minimo);
xa_t = (xa_t - minimo)./(maximo-minimo);
xb_t = (xb_t - minimo)./(maximo-minimo);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%sum e func de ativ%%%%%%%%%%%%%%%%%%%%%%%%%%

somatorio = xa.*w(1) + xb.*w(2);
somatorio = somatorio(1:1:s-1);
y=y(1:1:s-1);
yd=yd(1:1:s-1);

% for i=1:1:s-1  
%     y(i) = sign(somatorio(i) - theta);
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%calculo erro e delta rule%%%%%%%%%%%%%%%%%%%
% 
%  erro  = (yd - y);
% 
%  delta1 = alpha*erro*xa';
%  delta2 = alpha*erro*xb';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%treino%%%%%%%%%%%%%%%%%%% 
while 1 
    
    for j=1:1:s-1 
    percepton_output = tanh(xa(j)*w(1)+xb(j)*w(2)-theta*w(3));
    
    w(1) = w(1) + alpha*(yd(j)-percepton_output)*xa(j);
    w(2) = w(2) + alpha*(yd(j)-percepton_output)*xb(j);
    w(1) = w(1) + alpha*(yd(j)-percepton_output)*theta;
    
    end
    y = tanh(xa.*w(1)+xb.*w(2)-theta*w(3))
    if(abs(yd-y) <= erro_desejado)
        break
    end
end


%%plot(iteracao,vetor_erro);

%%%%%%%%%%%%%%%%%%%%%%%testar%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_t=y_t(1:1:length(B));
y_t = tanh(xa_t.*w(1) + xb_t.*w(2) -theta*(w(3)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 