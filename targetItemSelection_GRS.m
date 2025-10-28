function [targetItems] = targetItemSelection_GRS(data,otherItems,metric,attackIntent)
threshold=3;
if strcmp(attackIntent,'PUSH')==1

    if strcmp(metric,'MUP_Sum')==1
        toplam_Deger=sum(data);
        for i =1:size(toplam_Deger,2)
            for j=1:size(otherItems,2)
                if (i==otherItems(j))
                    toplam_Deger(1,i)=size(data,2)*5;
                end
            end
        end
        [degerSum, indisSum]=sort(toplam_Deger,'ascend');
        targetItems=indisSum;
    end

    if strcmp(metric,'MUP_Nnz')==1
        for k=1:size(data,2)
            numberOfNNZ(1,k)=nnz(data(:,k));
        end
        for i=1:size(data,2)
            for j=1:size(otherItems,2)
                if (i==otherItems(j))
                    numberOfNNZ(1,i)=size(data,1)+1;
                end
            end
        end
        [degerNnz, indisNnz]=sort(numberOfNNZ,'ascend');
        targetItems=indisNnz;
    end

    if strcmp(metric,'MUP_Th')==1
        for k=1:size(data,2)
           up_Threshold= size(find(data(:,k)>3),1);
           down_Threshold=size(find(data(:,k)<=3 & data(:,k)~=0),1);
           sizeThreshold(1,k)=up_Threshold/down_Threshold;
        end
        for i=1:size(data,2)
            for j=1:size(otherItems,2)
                if (i==otherItems(j))
                    sizeThreshold(1,i)=size(data,1)+1;
                end
            end
        end
        [degerThresh, indisThresh]=sort(sizeThreshold,'ascend');
        targetItems=indisThresh;
    end

    if strcmp(metric,'MUP_Ratio')==1
        for k=1:size(data,2)
           down_Threshold=size(find(data(:,k)<=3 & data(:,k)~=0),1);
           ratioThreshold(1,k)=down_Threshold/(nnz(data(:,k)));
        end
        for i=1:size(data,2)
            for j=1:size(otherItems,2)
                if (i==otherItems(j))
                    ratioThreshold(1,i)=1-size(data,1);
                end
            end
        end
        [degerRatio, indisRatio]=sort(ratioThreshold,'descend');
        targetItems=indisRatio;
    end
end
end
