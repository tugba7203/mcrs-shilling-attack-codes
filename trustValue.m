function [trustData] = trustValue(data)
for i=1:size(data,1)
    for j=1:size(data,1)
        if i==j
            trustData(i,j)=0;
        else
            distance=1/(1+(sqrt(sum(pow2((data(i,intersect(find(data(i,:)~=0),find(data(j,:)~=0)))-data(j,intersect(find(data(i,:)~=0),find(data(j,:)~=0)))))))));
            %                         jaccard=(size(intersect(find(data(i,:)~=0),find(data(j,:)~=0)),2))/(size(find(data(i,:)~=0),2)+size(find(data(j,:)~=0),2)-size(intersect(find(data(i,:)~=0),find(data(j,:)~=0)),2));
            %              jaccard1=(size(intersect(find(data(i,:)~=0),find(data(j,:)~=0)),2))/(size(find(data(i,:)~=0),2));
            commonRated=intersect(find(data(i,:)~=0),find(data(j,:)~=0));
            jaccard2=(size(intersect(find(data(i,:)>=3),find(data(j,:)>=3)),2))/(size(find(data(i,:)>=3),2));
            jaccard2(isnan(jaccard2))=0;
            jaccard3=(size(intersect(find(data(i,commonRated)<3),find(data(j,commonRated)<3)),2))/(size(commonRated,2));
            jaccard3(isnan(jaccard3))=0;
            jaccard=jaccard2+jaccard3;
            trustData(i,j)=(2*jaccard*distance)/(jaccard+distance);
        end
    end
end
end