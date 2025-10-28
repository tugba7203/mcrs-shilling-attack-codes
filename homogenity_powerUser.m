function [powerUser_dist] = homogenity_powerUser(powerUserAll,similarityAll,clusterSize,ratio,data,secim)

if strcmp(secim,'Sim')==1
for i=1:size(powerUserAll,2)
    powerData=powerUserAll{1,i};
    similarity=similarityAll{1,i};
    powerUser=powerData;%powerData(1:(size(powerData,1)*ratio));
    attackPowerUser_data=data(powerUser,:);
    firstPU=powerUser(1);
    [M,I]=(min(similarity(firstPU,powerUser)));
    secondPU=powerUser(I);
    diff_PU=[firstPU;secondPU];
    for j=1:(size(powerUser,1)-2)
        for k=1:size(diff_PU,1)
        diff_PUSim(k,:)= similarity(diff_PU(k,1),powerUser);
        end
        sumDiff_PU=sum(diff_PUSim);
        [value_SumSim indis_SumSim]=sort(sumDiff_PU,'ascend');
        if value_SumSim(1,1)<=0
        otherPU=powerUser(indis_SumSim');
        for l=1:size(diff_PU,1)
            [C Indis]=find(diff_PU(l,1)==otherPU);
            otherPU(C)=[];
        end
        diff_PU=[diff_PU;otherPU(1)];
        else 
            continue;
        end
      
    end
powerUser_dist{1,i}=diff_PU;
end
else
    ratio=0.1;
 for i=1:size(powerUserAll,2)
    powerData=powerUserAll{1,i};
    similarity=similarityAll{1,i};
    powerUser=powerData;%powerData(1:(size(powerData,1)*ratio));
    attackPowerUser_data=data(powerUser,:);
    firstPU=powerUser(1);
    powerUser(1)=[];
    [M,I]=(min(similarity(firstPU,powerUser)));
    secondPU=powerUser(I);
    powerUser(I)=[];
    diff_PU=[firstPU;secondPU];
    for j=1:(size(powerUser,1)-2)
        for k=1:size(diff_PU,1)
        diff_PUSim_a(k,:)= similarity(diff_PU(k,1),powerUser);
        diff_PUSim_b(k,:)= similarity(powerUser,diff_PU(k,1));
        diff_PUSim(k,:)=(diff_PUSim_a(k,:)+diff_PUSim_b(k,:))/2;
        end
        for h=1:size(diff_PUSim,2)
            numberOfNonZero(1,h)=nnz(diff_PUSim(:,h))/size(diff_PUSim,1);
            if numberOfNonZero(1,h)==0
                sumDiff_PU(1,h)=0;
            else
                sumDiff_PU(1,h)=sum(diff_PUSim(:,h))/nnz(diff_PUSim(:,h));
            end
        end
        sumDifferentUsers=sumDiff_PU+numberOfNonZero;
        [value_SumSim indis_SumSim]=sort(sumDifferentUsers,'ascend');
        if value_SumSim(1,1)<1
        otherPU=powerUser(indis_SumSim');
        for l=1:size(diff_PU,1)
            [C Indis]=find(diff_PU(l,1)==otherPU);
            otherPU(C)=[];
        end
        diff_PU=[diff_PU;otherPU(1)];
        else 
            continue;
        end
      
    end
powerUser_dist{1,i}=diff_PU(1:size(diff_PU,1)*ratio,1);   
 end
end
end

    %% En çok iki tane rated edenler
%     powerUsers=attackPowerUser_data(1,:);
%     sayac=1;
%     for k=2:size(attackPowerUser_data,1)
% %         powerUsers=attackPowerUser_data(k,:);
%         for t=1:size(attackPowerUser_data,2)
%             if(nnz(powerUsers(:,t))>2)
%                 break;
%             else
%                 if(t==size(attackPowerUser_data,2))
%                     powerUserID(sayac,1)=powerUser(k);
%                     powerUsers=[powerUsers;attackPowerUser_data(k,:)];
%                     sayac=sayac+1;
%                 end
%             end
% 
%         end
%     end

%%