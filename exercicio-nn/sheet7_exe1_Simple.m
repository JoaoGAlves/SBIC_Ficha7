clf;

syms x;

xa = zeros(1,1000);
xb = xa; %inicializar xa e xb
yd = xb;
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
w(3) = -0.5 + 0.5*rand(1,1);
alpha = 0.05;
theta = 1;

%%%%%%%%%%%%%%%%ler txt%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sizeA = [3 1000];
sizeB = [2 1000];
sizeC = [1 1000];

f = 3;

switch f
    case 1
        stringInputF = 'testInput10A.txt';
        stringOutputF = 'testOutput10A.txt';
    case 2
        stringInputF = 'testInput10B.txt';
        stringOutputF = 'testOutput10B.txt';
    case 3
        stringInputF = 'testInput10C.txt';
        stringOutputF = 'testOutput10C.txt';
end
fileID = fopen(stringInputF, 'r');
    [A,count] = fscanf(fileID, '%f,%f,%d', sizeA); %lê tudo que é para treino
    B=[A(1,length(A)); A(2,length(A))];
    [B1, count] = fscanf(fileID, '%f,%f', sizeB);
    B=[B(1,1), B1(1,:);B(2,1), B1(2,:)]; %lê tudo que é para testar
fclose(fileID);

fileID = fopen(stringOutputF, 'r');
    [C,count] = fscanf(fileID, '%d', sizeC);
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
           R(i) = 1; %plot blue (1)
       else
           R(i) = 0; %plot  red (-1)
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
    figure(1)
    hold on
    plot(xa.*R,xb.*R,'ro'); %red = 1
    plot(xa.*(1-R),xb.*(1-R),'bo'); %blue = -1
    title('Plot dos pontos (xa,xb) lidos do ficheiro')
    xlabel('xa')
    ylabel('xb')
    hold off
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
    %plot(xa,2.2594*xa-0.54); %linha
    title('Plot dos pontos (xa,xb) normalizados')
    xlabel('xa')
    ylabel('xb')
    hold off
    
%%%%%%%%%%%%%%%%sum e func de ativ%%%%%%%%%%%%%%%%%%%%%%%%%%

y=y(1:1:s-1);
yd=yd(1:1:s-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%treino%%%%%%%%%%%%%%%%%%% 
while 1 
    
    for j=1:1:s-1 
    percepton_output = sign(xa(j)*w(1)+xb(j)*w(2)+theta*w(3));
    
    w(1) = w(1) + alpha*(yd(j)-percepton_output)*xa(j);
    w(2) = w(2) + alpha*(yd(j)-percepton_output)*xb(j);
    w(3) = w(3) + alpha*(yd(j)-percepton_output)*theta;
    
    end
    y = sign(xa.*w(1)+xb.*w(2)+theta*w(3));
    if(yd-y == 0)
        break
    end
end


%%%%%%%%%%%%%%%%%%%%%%%testar%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_t=y_t(1:1:length(B));
y_t = sign(xa_t.*w(1) + xb_t.*w(2) + theta*(w(3)));
comparacao_entre_resultado_e_target = C - y_t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 