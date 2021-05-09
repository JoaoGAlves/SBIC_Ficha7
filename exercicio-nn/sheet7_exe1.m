clf;

xa = zeros(1,1000);
xb = xa; %inicializar xa e xb
yd = xb;
somatorio= yd;
y = yd;

w = rand(1,2); % wa = w(1) ...
alpha = 0.3;
theta = 0.5;

%%%%%%%%%%%%%%%%ler txt%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sizeA = [3 1000];
fileID = fopen('testInput10A.txt', 'r');

[A,count] = fscanf(fileID, '%f,%f,%d', sizeA); %lê tudo que é para treino
fclose(fileID);
A=A';
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xa = xa(1:1:s-1); %truncar 
xb = xb(1:1:s-1);
%%%%%%%%%%%%%%%%normalizar dados de input%%%%%%%%%%%%%%%%%%%
max = 100;
min = -100;
xa = (xa - min)./(max-min);
xb = (xb - min)./(max-min);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%sum e func de ativ%%%%%%%%%%%%%%%%%%%%%%%%%%
b=0; %Orde Orig
m=1; %declive

somatorio = xa.*w(1) + xb.*w(2);
somatorio = somatorio(1:1:s-1);
for i=1:1:s-1  
    y(i) = m*(somatorio(i) - theta) + b ;
end
y=y(1:1:s-1);
yd=yd(1:1:s-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%calculo erro e delta rule%%%%%%%%%%%%%%%%%%%
erro = (yd(1) - y(1) );
delta1 = alpha*erro.*xa(1);
delta2 = alpha*erro.*xb(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%treino%%%%%%%%%%%%%%%%%%%
for i=1:1:s-1 
    while 1 
        
        w(1) = w(1) + delta1;
        w(2) = w(2) + delta2;
        
        somatorio(i) = xa(i).*w(1) + xb(i).*w(2);
        y(i) = m*(somatorio(i) - theta) + b ; 
        
        erro = (yd(i) - y(i) );    
        delta1 = alpha*erro*xa(i);
        delta2 = alpha*erro*xb(i); 
        
        if(abs(erro) < 0.001)
            break;
        end
               
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 