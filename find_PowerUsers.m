function [InDegree_S,InDegree_A,InDegree_D,InDegree_V ,InDegree_O] = find_PowerUsers(similarityAll,Train_All,Power_N,cValue )
[satir sutun]=size(similarityAll);
for i=1:sutun
    sim_dataSet=similarityAll{1,i};
    trainData=Train_All{1,i};
    [satirSim sutunSim]=size(sim_dataSet);
    [satirTrain sutunTrain]=size(trainData);
    sayac=1;
    %%---------------------******-------------------
    if strcmp(cValue,'ID')==1
    for j=1:satirSim    
    [B I]=sort(sim_dataSet(j,:),'descend');
    I(sort(find(isnan(B)),'descend'))=[];
    B(sort(find(isnan(B)),'descend'))=[];
    if isempty(I)
        continue;
    else
    Komsu{1,sayac}=sort(I(1:Power_N),'ascend');
    KomsuMatris(sayac,:)=sort(I(1:Power_N),'ascend');
    sayac=sayac+1;
    end
    end
    [ii,jj,kk]=unique(cell2mat(Komsu),'stable');
    out{1,i}=ii(histc(kk,1:numel(ii))>1);
    end
    %%---------------------******-------------------
    if strcmp(cValue,'AS')==1
        sim_Data=similarityAll{1,i};   
    for j=1:satirSim
         [val ind]=find(isnan(sim_Data(j,:)));
         sim_Data(j,ind)=0;
        aggregate_sim(j,1)=sum(sim_Data(j,:));
    end
    [B I]=sort(aggregate_sim,'descend');
    I(sort(find(isnan(B)),'descend'))=[];
    B(sort(find(isnan(B)),'descend'))=[];
    out{1,i}=I;%(1:Power_N);
    end
    %%---------------------******-------------------
    if strcmp(cValue,'NR')==1
    for j=1:satirTrain
       countNR(1,j)=nnz(trainData(:,j)); 
    end
    [B I]=sort(countNR,'descend');
    I(sort(find(isnan(B)),'descend'))=[];
    B(sort(find(isnan(B)),'descend'))=[];
    out{1,i}=I;%(1:Power_N);
    end
    %%---------------------******-------------------
    if strcmp(cValue,'SR')==1
        sim_Data=similarityAll{1,i};   
    for j=1:satirSim
         [valP indP]=find((sim_Data(j,:))>0);
         positive_Sim=sum(sim_Data(j,indP(:)));
         [valN indN]=find((sim_Data(j,:))<0);
         negative_Sim=sum(sim_Data(j,indN(:)));
         sim_Ratio(j,1)=positive_Sim/abs(negative_Sim);
    end
    [B I]=sort(sim_Ratio,'descend');
    I(sort(find(isnan(B)),'descend'))=[];
    B(sort(find(isnan(B)),'descend'))=[];
    out{1,i}=I; %(1:Power_N);
    end

    if strcmp(cValue,'TR')==1
        [trustData] = trustValue(trainData);
    for j=1:size(trustData,1)
         aggregate_trust(j,1)=sum(trustData(j,:));
    end
    [B I]=sort(aggregate_trust,'descend');
    I(sort(find(isnan(B)),'descend'))=[];
    B(sort(find(isnan(B)),'descend'))=[];
    out{1,i}=I;%(1:Power_N);
    end
  
end

InDegree_S=out{1,1};
InDegree_A=out{1,2};
InDegree_D=out{1,3};
InDegree_V=out{1,4};
InDegree_O=out{1,5};
end



