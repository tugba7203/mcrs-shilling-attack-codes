function [GroupRating] = mrp_Aggregation(GrupKM)
mean_Data=mean(GrupKM);
for i=1:size(GrupKM,1)
fark_User(i)=sum(GrupKM(i,:)-mean_Data);
end
[valueUser indisUser]=sort(fark_User,'descend');
GroupRating=GrupKM(indisUser(1),:)
end