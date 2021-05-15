clf;

syms y_deriv

xa = zeros(1,1000);
xb = xa; %inicializar xa e xb
yd = xb;
y = yd;
xa_t=xa;
xb_t=xb;
y_t = yd;
vetor_erro = y_t;
R = yd;

deltak = zeros(0,0);

erro_desejado = ones(1,200).*0.15;
w=zeros(2,2);
w(:,:,1)=[w];
w(:,:,2) =[w];

%w(1,1,1) = -0.5 + 0.5*rand(1,1); %primeiras linhas
%w(1,2,1) = -0.5 + 0.5*rand(1,1);
%w(2,1,1) = -0.5 + 0.5*rand(1,1);
%w(2,2,1) = -0.5 + 0.5*rand(1,1);
w(1,1,1) = 0.3;
w(1,2,1) = 0.3;
w(2,1,1) = 0.3;
w(2,2,1) = 0.3;



%w(1,1,2) = -0.5 + 0.5*rand(1,1); %segundas linhas
%w(2,1,2) = -0.5 + 0.5*rand(1,1);
w(1,1,2) = 0.3;
w(2,1,2) = 0.3;

%w(:,:,:)
alpha = 0.5;
theta = 0.5;
b = ones(1,2).*2;
b2 = ones(1,2).*2;

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
    [A,count] = fscanf(fileID, '%f,%f,%d', sizeA); %lÃª tudo que Ã© para treino
    B=[A(1,length(A)); A(2,length(A))];
    [B1, count] = fscanf(fileID, '%f,%f', sizeB);
    B=[B(1,1), B1(1,:);B(2,1), B1(2,:)]; %lÃª tudo que Ã© para testar
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
maximo_xa = max(xa);
minimo_xa = min(xa);
maximo_xb = max(xb);
minimo_xb = min(xb);
xa = (xa - minimo_xa)/(maximo_xa-minimo_xa);
xb = (xb - minimo_xb)/(maximo_xb-minimo_xb);
yd = (yd - -1)./(1 - - 1);
xa_t = (xa_t - minimo_xa)./(maximo_xa-minimo_xa);
xb_t = (xb_t - minimo_xb)./(maximo_xb-minimo_xb);
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
erro_desejado = erro_desejado(1:1:s-1);
y=y(1:1:s-1);
yd=yd(1:1:s-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%treino%%%%%%%%%%%%%%%%%%% 
while 1 
    
    for j=1:1:s-1 
            %forward prop
         %xa(j)
         %xb(j)
         h(1)=w(1,1,1)*xa(j)+w(2,1,1)*xb(j) - b(1); %w*x = h
         out_h(1)=1/(1+exp(-h(1))); %devia de levar for, mas ta hardcoded
         h(2)=w(1,2,1)*xa(j)+w(2,2,1)*xb(j) - b(2);
         out_h(2)=1/(1+exp(-h(2))); %%out de h
         
         for k=1:1:1 %outputs
             out=w(1,k,2)*out_h(1)+w(2,k,2)*out_h(2) - b2(1)
             y(j)=1/(1+exp(-out)) %output Y
         end
         for k=1:1:1
            E_tot = (1/2)*(yd(j)-y(j))^2;
         end
           %começar daqui a implementar backpropagation...............
           %back prop
           
         for k=1:1:2
             deltak = y(j)*(1-y(j))*(yd(j)-y(j));
         end
         
         for i=1:1:2
             for k=1:1:2
                 
                 if (k == 1)
                        x = xa(j);
                 end
                 if(k == 2)
                        x = xb(j);
                 end
                 
                deltah = out_h(i)*(1-out_h(i))*(w(i,1,2)*deltak);
                w(k,i,1) = w(k,i,1) + alpha*deltah*x;
             end
         end
         
         for k=1:1:2
             w(k,1,2) = w(k,1,2) + deltak*alpha*out_h(k);
         end
                   
    end
    
    if(abs(yd-y) <= erro_desejado)
        break
    end
end






