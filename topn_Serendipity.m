function [ novelty_pri,diversity_pri] = topn_Serendipity( topn_idx_pri,sim_all_item,data_all_train,N ,rated)
S_data=data_all_train{1,1};
% [~, ~]=size(S_data);
min_val_pri=zeros();
% NOVELTY
sum_nov_pri=0;
for o=1:4
    data=data_all_train{1,o};    
for i=1:N  
    if i>size(topn_idx_pri,2)
        continue;
    else
    p2=nnz(data(:,(topn_idx_pri(1,i))))/size(S_data,1);
    if (isnan(p2))
        p2=1;
    end
    sum_nov_pri=sum_nov_pri-log2(p2);
    end
end
end
novelty_pri=sum_nov_pri/((4*N)*(-log2(1/size(S_data,1))));
% DIVERSITY
sum_div_pri=0;
for o=1:4
    sim=sim_all_item{1,o};
    sim(isnan(sim))=0;
for i=1:(N-1)
    for j=(i+1):N        
        if ((i>size(topn_idx_pri,2)) || (j>size(topn_idx_pri,2) ))
        continue;
        else
        cc2=(1-sim((topn_idx_pri(1,i)),(topn_idx_pri(1,j))));
        if (isnan(cc2))
        cc2=0;
        end
        sum_div_pri=sum_div_pri+cc2;
        end
    end
end
end
diversity_pri=sum_div_pri/(4*(N)*(N-1));
% SERENDIPITY  
%     sum_rs_pri=0;
% for o=1:4
%     data=data_all_train{1,o};
% for i=1:N      
%         if i>size(topn_idx_pri,2)
%             hedef_item_pri=[];
%         else
%         hedef_item_pri=find(data(:,(topn_idx_pri(1,i)))~=0);
%         end
%      epsilon=0.00000001;
%         for j=1:size(rated,2)
%         rated_item=find(data(:,rated(1,j))~=0);
%         [idxA,~] = find(bsxfun(@eq,hedef_item_pri,rated_item.'));
%         CC=sort(hedef_item_pri(idxA),'ascend');
%         PMI_pri=(log2((length(CC)/size(S_data,1)+epsilon)/((length(hedef_item_pri)/size(S_data,1))*(length(rated_item)/size(S_data,1)))))/(-log2(length(CC)/size(S_data,1)));
%         if (isnan(PMI_pri))
%         PMI_pri=0;
%         end
%         min_val_pri(1,j)=(1-PMI_pri)/2;
%         end
%     sum_rs_pri=sum_rs_pri+min(min_val_pri(:));
% end
% end
% Serendipity_pri=sum_rs_pri/(4*N);
end

