clf;

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
w=zeros(2,2);
w(:,:,1)=[w];
w(:,:,2) =[w];

w(1,1,1) = -0.5 + 0.5*rand(1,1); %primeiras linhas
w(1,2,1) = -0.5 + 0.5*rand(1,1);
w(2,1,1) = -0.5 + 0.5*rand(1,1);
w(2,2,1) = -0.5 + 0.5*rand(1,1);

w(1,1,2) = -0.5 + 0.5*rand(1,1); %segundas linhas
w(2,1,2) = -0.5 + 0.5*rand(1,1);

%w(:,:,:)
alpha = 0.05;
theta = 1;
b = zeros(1,2);
b2 = zeros(1,2);

%%%%%%%%%%%%%%%%ler txt%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sizeA = [3 1000];
sizeB = [2 1000];
sizeC = [1 1000];

f = 1;

switch f
    case 1
        stringInputF = 'testInput11A.txt';
        stringOutputF = 'testOutput11A.txt';
    case 2
        stringInputF = 'testInput11B.txt';
        stringOutputF = 'testOutput11B.txt';
    case 3
        stringInputF = 'testInput11C.txt';
        stringOutputF = 'testOutput11C.txt';
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

         h(1)=sigmoid(w(1,1,1)*xa(j)+w(2,1,1)*xb(j)+b(1)) %devia de levar for, mas ta hardcoded
         h(2)=sigmoid(w(1,2,1)*xa(j)+w(2,2,1)*xb(j)+b(2))
         for k=1:1:1
             y(j)=sigmoid(w(1,k,2)*h(1)+w(2,k,2)*h(2)+b2(1));
         end
         for k=1:1:1
            E_tot = (1/2)*(yd(j)-y(j))^2
         end
           %come�ar daqui a implementar backpropagation...............
    end
   
    
    y = sign(xa.*w(1)+xb.*w(2)+theta*w(3));
    if(yd-y == 0)
        break
    end
end






