function [fbest,sbest,fbest_hist,search_hist,ave_fit,x_1st, Exploration, Exploitation]=CCO(pop,iter,lb,ub,dim,fobj);


lb=lb.*ones(1,dim);
ub=ub.*ones(1,dim);
x =initialization(pop,dim,ub,lb);
x_new=zeros(pop,dim);

fbest=inf;
sbest=zeros(1,dim);
fitness=zeros(pop,1);
fitness_new=zeros(pop,1);
fbest_hist=ones(1,iter);
search_hist=zeros(pop*iter,dim);
ave_fit=zeros(1,iter);
x_1st=zeros(1,iter);
diversity=zeros(1,iter);
Exploration=zeros(1,iter);
Exploitation=zeros(1,iter);

for i = 1:pop
    fitness(i)=fobj(x(i,:));
    if fitness(i)<fbest
        fbest=fitness(i);
        sbest=x(i,:);
    end
end

[~,class]=sort(fitness);
x=x(class,:);
KP=randperm(pop,pop/2);
K=0.5;

for i = 1 : iter
        x_mean=mean(x);
        x_c=x-x_mean;
        C=cov(x_c(pop/2:pop,:));
        [V,D]=eig(C);
        x_pca=x*V;
        a1=1;
        a2=1;
    for j =1:pop
        t=randperm(pop,2);
        p=rand;
        U=rand(1,dim);
        alpha=(cos(i*pi/iter)+1)/3+0.2;
        U=U>alpha;
        if rand<1
            r=rand*1;
        else
            r=randn(1,dim)*1;
        end
        r1=alpha;
        if ismember(j, KP)
            x_newpca(j,:)=U.*x_pca(randi(j),:)+(1-U).*(x_pca((j),:)*r1+x_pca(randi(j),:)*(1-r1)+r.*(x_pca(randi(j),:)-x_pca(randi(pop),:)+(1-r).*(x_pca(randi(j),:)-x_pca(randi(pop),:))));
            x_new(j,:)=x_newpca(j,:)*V';
            for k =1 :dim
               if x_new(j,k)<lb(k) 
                  x_new(j,k)=lb(k)+rand*(ub(k)-lb(k));
               elseif x_new(j,k)>ub(k)
                  x_new(j,k)=lb(k)+rand*(ub(k)-lb(k));
               end
            end
            fitness_new(j)=fobj(x_new(j,:));
            if fitness_new(j)<fitness(j)
                x(j,:)=x_new(j,:);
                fitness(j)=fitness_new(j);
                a1=a1+1;
            end
        else
            x_new(j,:)=U.*x(randi(j),:)+(1-U).*(x((j),:)*r1+x(randi(j),:)*(1-r1)+r.*(x(randi(j),:)-x(randi(pop),:)+(1-r).*(x(randi(j),:)-x(randi(pop),:))));
            for k =1 :dim
               if x_new(j,k)<lb(k) 
                  x_new(j,k)=lb(k)+rand*(ub(k)-lb(k));
               elseif x_new(j,k)>ub(k)
                  x_new(j,k)=lb(k)+rand*(ub(k)-lb(k));
               end
            end
            fitness_new(j)=fobj(x_new(j,:));
            if fitness_new(j)<fitness(j)
                x(j,:)=x_new(j,:);
                fitness(j)=fitness_new(j);
                a2=a2+1;
            end
        end
    end
    K=a1/K/(a1/K+a2/(1-K));
    if K<0.3
        K=0.3;
    elseif K>0.7
        K=0.7;
    end
    kp=round(K*pop);
    KP=randperm(pop,kp);   

    [fitness,class]=sort(fitness);
    x=x(class,:);
    ave_fit(i)=mean(fitness);
    x_1st(i)=x(1,1);
    
    diver=zeros(1,dim);
    for k=1:dim
        div=zeros(1,pop);
        for j=1:pop
             div(j)=abs(median(x(:,k))-x(j,k));
        end
        diver(k)=sum(div(j))/pop;
    end
    diversity(i)=sum(diver)/dim;

    if fitness(1)<fbest
       fbest=fitness(1);
       sbest=x(1,:);
    end
    fbest_hist(i)=fbest;
    search_hist(((i-1)*pop+1:i*pop),:)=x;
end
    Exploration=diversity/max(diversity)*100;
    Exploitation=(1-diversity/max(diversity))*100;
end

function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= length(ub); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
     Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
         Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;      
    end
end

end
