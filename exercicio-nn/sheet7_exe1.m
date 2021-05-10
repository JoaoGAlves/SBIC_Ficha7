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
R = yd;

erro_desejado = ones(1,200).*0.15;

cnt = 1;

w(1) = -0.5 + 0.5*rand(1,1);
w(2) = -0.5 + 0.5*rand(1,1);
alpha = 0.3;
theta = 0;

%%%%%%%%%%%%%%%%ler txt%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sizeA = [3 1000];
sizeB = [2 1000];
fileID = fopen('testInput10A.txt', 'r');

[A,count] = fscanf(fileID, '%f,%f,%d', sizeA); %l� tudo que � para treino

B=[A(1,length(A)); A(2,length(A))];
[B1, count] = fscanf(fileID, '%f,%f', sizeB);
B=[B(1,1), B1(1,:);B(2,1), B1(2,:)]; %l� tudo que � para testar
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
       if(A(i,3) == 1)
           R(i) = 1; %plot red
       else
           R(i) = 0;
       end
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
R = R(1:1:s-1);
vetor_erro = vetor_erro(1:1:s-1);
figure(1)
    hold on
    plot(xa.*R,xb.*R,'ro'); %red = 1
    plot(xa.*(1-R),xb.*(1-R),'bo'); %blue = -1
    title('Plot dos pontos (xa,xb) lidos do ficheiro')
    xlabel('xa')
    ylabel('xb')
    hold off
    %xb.*R;

%%%%%%%%%%%%%%%%normalizar dados de input%%%%%%%%%%%%%%%%%%%
maximo = max([xa,xb]);
minimo = min([xa,xb]);
xa = (xa - minimo)./(maximo-minimo);
xb = (xb - minimo)./(maximo-minimo);
xa_t = (xa_t - minimo)./(maximo-minimo);
xb_t = (xb_t - minimo)./(maximo-minimo);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
figure(2)
    hold on
    plot(xa.*R,xb.*R,'ro'); %red = 1
    plot(xa.*(1-R),xb.*(1-R),'bo'); %blue = -1
    plot(xa,2.2594*xa-0.54); %linha
    title('Plot dos pontos (xa,xb) normalizados')
    xlabel('xa')
    ylabel('xb')
    hold off
    %xb.*R;
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
    y(j) = tanh(somatorio(j) - theta);
    end

    erro  = (yd - y);

    delta1 = alpha*erro*xa';
    delta2 = alpha*erro*xb';
    
    w(1) = w(1) + delta1;
    w(2) = w(2) + delta2;
    
    somatorio = xa.*w(1) + xb.*w(2);

    if(cnt > 50000)
        break;
    end
    cnt = cnt +1;        
end

iteracao = linspace(0,cnt,cnt);

%%plot(iteracao,vetor_erro);

%%%%%%%%%%%%%%%%%%%%%%%testar%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_t=y_t(1:1:length(B));
somatorio_t = xa_t.*w(1) + xb_t.*w(2);
somatorio_t = somatorio_t(1:1:length(B));
for i=1:1:length(B)
    y_t(i) = tanh(somatorio_t(i) - theta);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 