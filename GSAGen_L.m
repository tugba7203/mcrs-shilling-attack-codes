function [random_AttackData,average_AttackData] = GSAGen_L(Train_All,filler_size_ratio,attack_size,multiple_targetItems)

for i=1:size(Train_All,2)
    data=Train_All{1,i};
    attack_data=zeros(round(size(data,1)*attack_size),round(size(data,2)));
    attack_data2=zeros(round(size(data,1)*attack_size),round(size(data,2)));
    system_mean=mean(data(:));
    item_mean=mean(data);
    [satir sutun]=size(data);
    filler_size=round(sutun*filler_size_ratio)+size(multiple_targetItems,2);
    random_number=randperm(sutun,filler_size);
    count=0;
    for y=1:size(multiple_targetItems,2)
        [C I]=find(multiple_targetItems(1,y)==random_number);
        if isempty(I)
            %random_number(y)=[];
        else
            random_number(I)=[];
            count=count+1;
        end
    end
    %     if ((count<size(multiple_targetItems,2)) && (count>0))
    %         random_number(1:((size(multiple_targetItems,2)-count)))=[];
    %     end

    %     attack_data(1,random_number)=item_mean;
    attack_data(1,random_number)=system_mean;
    attack_data2(1,random_number)=item_mean(random_number);
    for k=2:round(satir*attack_size)
        random_number=randperm(sutun,filler_size);
        count=0;
        for y=1:size(multiple_targetItems,2)
            [C I]=find(multiple_targetItems(1,y)==random_number);
            if isempty(I)
                %random_number(y)=[];
            else
                random_number(I)=[];
            end
        end
        attack_data(k,random_number)=system_mean;
        attack_data2(k,random_number)=item_mean(random_number);
        for t=1:size(attack_data,2)
            %             ans(t)=nnz(attack_data(:,t))
            if(nnz(attack_data(:,t))>2)
                attack_data(k,t)=0;
                attack_data2(k,t)=0;
%                 for s=1:size(attack_data,2)
%                     if(nnz(attack_data(:,s))<2 && (attack_data(k,s)==0))
%                         attack_data(k,s)=system_mean;
%                         break;
%                     end
%                 end
            end
        end
    end
%     for t=1:size(attack_data2,2)
%         ans(t)=nnz(attack_data2(:,t))
%     end
    attack_data(:,multiple_targetItems)=max(data(:));
    attack_data2(:,multiple_targetItems)=max(data(:));
    random_AttackData{1,i}=[data;attack_data];
    average_AttackData{1,i}=[data;attack_data2];

end