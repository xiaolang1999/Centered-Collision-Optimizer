function [fbest,sbest,fbest_hist,search_hist]=CCO(pop,iter,lb,ub,dim,fobj);

%fbest represents the optimal fitness value in the output.
%sbest represents the optimal solution in the output.
%fbest_hist represents the history of the best fitness value at each iteration in the output.
%search_hist represents the set of all historical solutions in the output.
%pop represents the population size in the input.
%iter represents the maximum number of iterations in the output.
%lb represents the lower bound of the search space in the input.
%ub represents the upper bound of the search space in the input.
%dim represents the number of problem dimensions in the input.
%fobj represents the objective fitness function in the input.

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

    if fitness(1)<fbest
       fbest=fitness(1);
       sbest=x(1,:);
    end
    fbest_hist(i)=fbest;
    search_hist(((i-1)*pop+1:i*pop),:)=x;
end
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
