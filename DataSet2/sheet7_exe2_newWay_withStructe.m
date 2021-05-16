
clear all;
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
w=zeros(3,3);
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
w(3,1,1) = 0.3;
w(3,2,1) = 0.3;



%w(1,1,2) = -0.5 + 0.5*rand(1,1); %segundas linhas
%w(2,1,2) = -0.5 + 0.5*rand(1,1);
w(1,1,2) = 0.3;
w(2,1,2) = 0.3;



%w(:,:,:)
alpha = 0.15;

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

n_hidden_layers = 1;
n_nodes_per_layers = 5;

node(n_hidden_layers,n_nodes_per_layers).weights = zeros(1, n_nodes_per_layers);
node(n_hidden_layers,n_nodes_per_layers).bias = ones(1,n_nodes_per_layers);
node(n_hidden_layers,n_nodes_per_layers).outputA = zeros(1,s-1);
node(n_hidden_layers,n_nodes_per_layers).output = zeros(1,s-1);

for k=1:1:n_hidden_layers
    for i=1:1:n_nodes_per_layers
        node(k,i).weights = -0.5 + rand(1,n_nodes_per_layers);%-2.4/2 + -2.4/2*rand(1,n_nodes_per_layers);
        node(k,i).bias = -1;
    end
end

output_node.weights = -0.5 + rand(1,n_nodes_per_layers);%-2.4/2 + (2.4/2+2.4/2)*rand(1,n_nodes_per_layers);
output_node.bias = -1;
output_node.output = 0;
output_node.outputA = 0;
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
yd = (yd +1)./(1 + 1);
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

erro_j = zeros(1,5);
conta_it = 0;
while 1

%foward prop

%primeira hidden layer output
if( (yd-output_node.outputA) < 0.3*ones(1,s-1))
    break;
end
conta_it = conta_it +1
for j=1:1:s-1
    
    for m=1:1:n_nodes_per_layers
        output = 0;
        for k=1:1:2
             if (k == 1)
                    x = xa(j);
             end
             if(k == 2)
                    x = xb(j);
             end
            output = output + x*node(1,m).weights(k);
        end
        output = output - node(1,m).bias;
        node(1,m).output(j) = output;
        node(1,m).outputA(j) = 1/(1+exp(-node(1,m).output(j)));
    end
    
    %%foward para mais que uma hidden layer
    if(n_hidden_layers > 1)
        for i=2:1:n_hidden_layers %anda pelas hidden layers
            for m=1:1:n_nodes_per_layers %anda pelos nos de cada hidden layer
                output = 0;
                for k=1:1:n_nodes_per_layers % para ter os outputs das hidden layers anteriores e calcular outputs
                    %disp([i m k]);
                    output = output + node(i-1,k).outputA(j)*node(i,m).weights(k);
                end
                output = output - node(i,m).bias;
                node(i,m).output(j) = output;
                node(i,m).outputA(j) = 1/(1+exp(-node(i,m).output(j)));
            end
        end
    end
    
    %foward output
    
     for m=1:1:n_nodes_per_layers
        output = 0;
        k=0;
        for k=1:1:n_nodes_per_layers
            output = output + node(n_hidden_layers,k).outputA(j)*output_node.weights(k);
        end
        output = output - output_node.bias;
        output_node.output(j) = output;
        output_node.outputA(j) = 1/(1+exp(-output_node.output(j)));
     end
    
     %--backward prop
     
     %delta do output
     deltaOut = (yd(j)-output_node.outputA(j))*output_node.outputA(j)*(1-output_node.outputA(j));
        
     %alterar se for preciso para usar o w antigos e nao os mudados
     delta_w_output = zeros(1,n_nodes_per_layers);
     for i=1:i:n_nodes_per_layers
        delta_w_output(i) = alpha*deltaOut*node(n_hidden_layers,i).outputA(j);
        output_node.weights(i) = output_node.weights(i) + delta_w_output(i);
     end
     
     somatorio = 0;
     if(n_hidden_layers == 1)
        for k=1:1:n_nodes_per_layers
            erro_j(k) = node(1,k).outputA(j)*(1-node(1,k).outputA(j));
            somatorio = somatorio + deltaOut*output_node.weights(k);
            erro_j(k) = erro_j(k)*somatorio;
        end
        %se der merda vem aqui ver
        for a=1:1:2 %numero inputs
            switch a
                case 1
                    x = xa(j);
                case 2
                    x = xb(j);
            end
            for b=1:1:n_nodes_per_layers
                node(1,b).weights(a) = node(1,b).weights(a) + alpha*erro_j(a)*x;
            
            end
        end
     end
end
end




