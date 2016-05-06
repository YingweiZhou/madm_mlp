%data labels
load data_batch_1.mat
data1=im2double(data')*255;
labels1=(labels);
load data_batch_2.mat
data2=im2double(data')*255;
labels2=(labels);
load data_batch_3.mat
data3=im2double(data')*255;
labels3=(labels);
load data_batch_4.mat
data4=im2double(data')*255;
labels4=(labels);
load data_batch_5.mat
data5=im2double(data')*255;
labels5=(labels);

images_total=[data1 data2 data3 data4 data5];
labels_total=[labels1;labels2;labels3;labels4;labels5];
%images_total=images_total(:,1:2000);
%labels_total=labels_total(1:2000);
[dimension,amount_total]=size(images_total);

%structure    
image_size=(dimension/3)^0.5;

wc=normrnd(0,1/image_size,dimension,400);

w1=normrnd(0,1/image_size,400,600+1);

w2=normrnd(0,1/image_size,600+1,10);

%normalization data
standard=mean(images_total,2);
bb=0;
for i=1:1:amount_total
    images_total(:,i)=(images_total(:,i)-standard(:,1));
    cc=sqrt(sum(images_total(:,i).^2));
    images_total(:,i)=images_total(:,i)/cc;    
    bb=bb+cc;
end
bb=bb/amount_total;
%parameters
epoc_rbm=70;
epoc_mlp=50*epoc_rbm;
power=2;    %first part
r=0.5;      %first part
lr=0.3;
sigma=10^-8;    %adadelta used in RMS(root mean square)
gamma=0.2;      %adadelta act as learning rate
minibatch=10;
momentum=0.3;
lambda=0.25;

%variable containers
errors_total=zeros(1,epoc_mlp);
interval=amount_total*0.1;
correct_predicted_amount=zeros(1,epoc_mlp);
correct_predicted_rate=zeros(1,epoc_mlp);
correct_predicted_rate_sum=zeros(1,epoc_mlp);
interval=amount_total*0.1;
%sig=1;
min_error=1;
%train first part
for j=1:epoc_rbm
           
    %shuffle
    cv_index=randperm(9);
    test=images_total(:,(interval*cv_index(9))+1:1:(interval*(cv_index(9)+1)));
    test_labels=labels_total(interval*cv_index(9)+1:1:(interval*(cv_index(9)+1)));
    if cv_index(9)==9
       train=images_total(:,1:1:interval*9);
       train_labels=labels_total(1:1:interval*9);
    end
    if cv_index(9)==1
       train=images_total(:,1*interval+1:1:amount_total);
       train_labels=labels_total(1*interval+1:1:amount_total);
    end
    if cv_index(9)>1 && cv_index(9)<9
        train=[images_total(:,1:interval*cv_index(9)) images_total(:,interval*(cv_index(9)+1)+1:amount_total)];
        train_labels=[labels_total(1:interval*cv_index(9)) ; labels_total(interval*(cv_index(9)+1)+1:amount_total)];
    end
     
    [train_dimension train_amount]=size(train);
    [test_dimension test_amount]=size(test);    
   
    batch_delta_wc=zeros(size(wc));    
    %train
    %momentum container
    %delta_wc_p=zeros(size(wc));
    wc_p=zeros(size(wc));
    for i=1:1:train_amount
        sprintf('trainning l in iteration %d of total epoc %d \n',i,j)             
        x=train(:,i);
        V=wc'*x;
        A = V.^power;
        A(find(V<0))=0;
        %weight updating
        %delta_wc=momentum*delta_wc_p-lr*(x-wc*A)*A';
        
        %delta_wc_p=delta_wc;
%        batch_delta_wc=batch_delta_wc+ delta_wc;        
        %update 
        wc=wc+lr*(x-wc*A)*A'+momentum*wc_p;
        wc_p=lr*(x-wc*A)*A'+momentum*wc_p;
%         if mod(i,minibatch)==0
%            wc=wc-(1/minibatch)*batch_delta_wc;
%            batch_delta_wc=zeros(size(wc));
%         end
    end
    
end
%draw part
col=ceil(sqrt(400));
row=ceil(400/col);
for i=1:400
    wa=wc(:,i);
    wa=wa-min(wa);
    wa=wa/max(wa);
    wa=wa*255;
    R=wa(1:1024);
    G=wa(1025:2048);
    B=wa(2049:3072);
    war(:,:,1)=reshape(R,32,32);
    war(:,:,2)=reshape(G,32,32);
    war(:,:,3)=reshape(B,32,32);
    war=uint8(war);
    subplot(row,col,i,'align');
    imshow(war);
end

%iteration
lr=0.3;
for iter=1:epoc_mlp
    %shuffle
    cv_index=randperm(9);
    test=images_total(:,(interval*cv_index(9))+1:1:(interval*(cv_index(9)+1)));
    test_labels=labels_total(interval*cv_index(9)+1:1:(interval*(cv_index(9)+1)));
    if cv_index(9)==9
       train=images_total(:,1:1:interval*9);
       train_labels=labels_total(1:1:interval*9);
    end
    if cv_index(9)==1
       train=images_total(:,1*interval+1:1:amount_total);
       train_labels=labels_total(1*interval+1:1:amount_total);
    end
    if cv_index(9)>1 && cv_index(9)<9
        train=[images_total(:,1:interval*cv_index(9)) images_total(:,interval*(cv_index(9)+1)+1:amount_total)];
        train_labels=[labels_total(1:interval*cv_index(9)) ; labels_total(interval*(cv_index(9)+1)+1:amount_total)];
    end
    %map train data with rbm structure
    V_output=wc'*train; %100*a
    train_output = V_output.^power;
    train_output(find(V_output<0))=0;
    
    [train_dimension train_amount]=size(train_output);
    [test_dimension test_amount]=size(test);    
    
    %train
    %container for previous value
%     batch_delta_w1=zeros(size(w1));
%     batch_delta_w2=zeros(size(w2));
%     batch_delta_b1=zeros(size(b1));
%     batch_delta_b2=zeros(size(b2));
    %momentum contianer
    delta1_p=zeros(size(w1));
    delta2_p=zeros(size(w2));
%     delta1_b_p=zeros(size(b1));
%     delta2_b_p=zeros(size(b2));
  w2_p=zeros(size(w2));
  w1_p=zeros(size(w1));
    for i=1:train_amount
        %Forward
        sprintf('training example %d in iteration %d current mimum error is %d',i,iter,min_error)
        x=train_output(:,i);
        label=train_labels(i)+1;

        z1=w1'*x;
        z1(1)=1;
        %f1=exp(z1)./(exp(sum(z1)));  %softmax
        f1=1./(1+exp(-z1));
        %layer1 to layer2
        x2=f1;
        z2=w2'*x2;
        
        %f2=exp(z2)./(sum(exp(z2)));  %softmax
        f2=1./(1+exp(-z2));
        %Backward
        cor=zeros(10,1);
        cor(label)=1;
        %quadratic error function sum((cor-f2).^2)/2;
        delta3=-0.2*((cor-f2).*(1-f2).*f2);
        delta2=(w2*delta3).*((1-f1).*f1);
        delta2_b=delta3;
        delta2_w=delta3*f1';
        delta1_w=delta2*x';
        delta1_b=delta2;
        
%         %batch updating with mini-batch
%         batch_delta_w1=batch_delta_w1+delta1_w';
%         batch_delta_w2=batch_delta_w2+delta2_w';
%         batch_delta_b1=batch_delta_b1+delta1_b;
%         batch_delta_b2=batch_delta_b2+delta2_b;
        %momenturm part
%         batch_delta_w1=batch_delta_w1+gamma*delta1_p+lr*(delta1_w');
%         batch_delta_w2=batch_delta_w2+gamma*delta2_p+lr*delta2_w';
%         batch_delta_b1=batch_delta_b1+gamma*delta1_b_p+lr*delta1_b;
%         batch_delta_b2=batch_delta_b2+gamma*delta2_b_p+lr*delta2_b;
%         delta1_p=gamma*delta1_p+lr*(delta1_w');
%         delta2_p=gamma*delta2_p+lr*delta2_w';
%         delta1_b_p=gamma*delta1_b_p+lr*delta1_b;
%         delta2_b_p=gamma*delta2_b_p+lr*delta2_b;
        %update if reaches size of batch
        w2=w2-lr*delta2_w'-momentum*w2_p;
        w1=w1-lr*(delta1_w')-momentum*w1_p;
        w2_p=lr*(delta2_w')+momentum*w2_p;
        w1_p=lr*(delta1_w')+momentum*w1_p;

%         if mod(i,minibatch)==0
%            w2=w2-lr*((1/minibatch)*batch_delta_w2+lambda*w2);
%            w1=w1-lr*((1/minibatch)*batch_delta_w1+lambda*w1);          
%            b1=b1-lr*(1/minibatch)*batch_delta_b1;
%            b2=b2-lr*(1/minibatch)*batch_delta_b2;
%            batch_delta_w1=zeros(size(w1));
%            batch_delta_w2=zeros(size(w2));
%            batch_delta_b1=zeros(size(b1));
%            batch_delta_b2=zeros(size(b2));
%         end
    end
    error_test=zeros(train_amount,1);
    for i=1:train_amount
        x=train(:,i);
        label=train_labels(i)+1;
        V_output=wc'*x; %100*a
        wc_output = V_output.^power;
        wc_output(find(V_output<0))=0;
        z1=w1'*wc_output;
        z1(1)=1;
        %f1=exp(z1)./(sum(exp(z1)));  %softmax
        f1=1./(1+exp(-z1));
        x2=f1;
        z2=w2'*x2;
        %f2=exp(z2)./(sum(exp(z2)));  %softmax
        f2=1./(1+exp(-z2));
        cor=zeros(10,1);
        cor(label)=1;
        %quadratic cost function
        error_test(i)=sum((cor-f2).^2)/10;
        [val idx]=max(f2);
        if idx==label
        correct_predicted_amount(iter)=correct_predicted_amount(iter)+1;
        end
%         correct_predicted_rate(iter)=correct_predicted_rate(iter)+f2(label);
%         correct_predicted_rate_sum(iter)=correct_predicted_rate_sum(iter)+f2(label)/sum(f2);
        sprintf('error_test in test example %d is %d',i,error_test(i))
    end
%     correct_predicted_rate(iter)=correct_predicted_rate(iter)/test_amount;
%     correct_predicted_rate_sum(iter)=correct_predicted_rate_sum(iter)/test_amount;
    errors_total(iter)=sum(error_test)/(train_amount);
    c=errors_total(iter);
    if c<min_error
        min_error=c;
    end
    sprintf('average prediction error in iteration %d is %d', iter,errors_total(iter))
    sprintf('correct_predicted_amount in iteration %d is %d',iter,correct_predicted_amount(iter))
%     sprintf('correct_predicted_rate in iteration %d is %d',iter,correct_predicted_rate(iter))
%     sprintf('correct_predicted_rate_sum in iteration %d is %d',iter,correct_predicted_rate_sum(iter))
end
    sprintf('average prediction error is %d', sum(errors_total)/epoc_mlp)
    sprintf('average correct_predicted_amount is %d', sum(correct_predicted_amount)/epoc_mlp)
%     sprintf('average correct_predicted_rate is %d',sum(correct_predicted_rate)/epoc)
%     sprintf('average correct_predicted_rate_sum is %d',sum(correct_predicted_rate_sum)/epoc)
    figure(1);
    plot(1:epoc_mlp,errors_total);
    hold on;
    figure(2);
    plot(1:epoc_mlp,errors_total);
    hold on;
    figure(3);
    plot(1:epoc_mlp,correct_predicted_amount);
    hold on;
%     figure(4);
%     plot(1:epoc,correct_predicted_rate);
%     hold on;
%     figure(5);
%     plot(1:epoc,correct_predicted_rate_sum);
    hold on;
    
    
    
%     
%      %test
%      error_test=zeros(test_amount,1);
%      for i=1:test_amount
%         x=test(:,i);
%         label=test_labels(i)+1;
%         V_output=wc'*x; %100*a
%         wc_output = V_output.^power;
%         wc_output(find(V_output<0))=0;
%         z1=w1'*wc_output;
%         z1(1)=1;
%         %f1=exp(z1)./(sum(exp(z1)));  %softmax
%         f1=1./(1+exp(-z1));
%         x2=f1;
%         z2=w2'*x2;
%         %f2=exp(z2)./(sum(exp(z2)));  %softmax
%         f2=1./(1+exp(-z2));
%         cor=zeros(10,1);
%         cor(label)=1;
%         %quadratic cost function
%         error_test(i)=sum((cor-f2).^2)/10;
%         [val idx]=max(f2);
%         if idx==label
%         correct_predicted_amount(iter)=correct_predicted_amount(iter)+1;
%         end
%         correct_predicted_rate(iter)=correct_predicted_rate(iter)+f2(label);
%         correct_predicted_rate_sum(iter)=correct_predicted_rate_sum(iter)+f2(label)/sum(f2);
%         sprintf('error_test in test example %d is %d',i,error_test(i))
%     end
%     correct_predicted_rate(iter)=correct_predicted_rate(iter)/test_amount;
%     correct_predicted_rate_sum(iter)=correct_predicted_rate_sum(iter)/test_amount;
%     errors_total(iter)=sum(error_test)/(test_amount);
%     c=errors_total(iter);
%     if c<min_error
%         min_error=c;
%     end
%     sprintf('average prediction error in iteration %d is %d', iter,errors_total(iter))
%     sprintf('correct_predicted_amount in iteration %d is %d',iter,correct_predicted_amount(iter))
%     sprintf('correct_predicted_rate in iteration %d is %d',iter,correct_predicted_rate(iter))
%     sprintf('correct_predicted_rate_sum in iteration %d is %d',iter,correct_predicted_rate_sum(iter))
% end
%     sprintf('average prediction error is %d', sum(errors_total)/epoc)
%     sprintf('average correct_predicted_amount is %d', sum(correct_predicted_amount)/epoc)
%     sprintf('average correct_predicted_rate is %d',sum(correct_predicted_rate)/epoc)
%     sprintf('average correct_predicted_rate_sum is %d',sum(correct_predicted_rate_sum)/epoc)
%     figure(1);
%     plot(1:epoc,errors_total);
%     hold on;
%     figure(2);
%     plot(1:epoc,errors_total);
%     hold on;
%     figure(3);
%     plot(1:epoc,correct_predicted_amount);
%     hold on;
%     figure(4);
%     plot(1:epoc,correct_predicted_rate);
%     hold on;
%     figure(5);
%     plot(1:epoc,correct_predicted_rate_sum);
%     hold on;

