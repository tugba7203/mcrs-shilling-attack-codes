function [similarity_Criteria] = similarity_measurePCC(Train_All,selection)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if strcmp(selection,'user')==1
if iscell(Train_All)==0
    dataSet=Train_All;
    for j=1:size(dataSet,1)
        mean_X=mean(dataSet(j,:));
        for k=1:size(dataSet,1)
            mean_Y=mean(dataSet(k,:));
            [coRatedValue,coRatedItem1,coRatedItem2]=intersect(find(dataSet(j,:)~=0),find(dataSet(k,:)~=0));
            numerator=sum((dataSet(j,coRatedValue)-mean_X).*(dataSet(k,coRatedValue)-mean_Y));
            deminator=sqrt(sum((dataSet(j,coRatedValue)-mean_X).^2))*sqrt(sum((dataSet(k,coRatedValue)-mean_Y).^2));
            similarity(j,k)=numerator/deminator;
        end
    end
    similarity_Criteria=similarity;
else
    for i=1:size(Train_All,2)
        similarity=[];
        dataSet=Train_All{1,i};
%         for j=1:size(dataSet,1)
%             mean_X=mean(dataSet(j,:));
%             for k=1:size(dataSet,1)
%                 mean_Y=mean(dataSet(k,:));
%                 [coRatedValue,coRatedItem1,coRatedItem2]=intersect(find(dataSet(j,:)~=0),find(dataSet(k,:)~=0));
%                 numerator=sum((dataSet(j,coRatedValue)-mean_X).*(dataSet(k,coRatedValue)-mean_Y));
%                 deminator=sqrt(sum((dataSet(j,coRatedValue)-mean_X).^2))*sqrt(sum((dataSet(k,coRatedValue)-mean_Y).^2));
%                 similarity(j,k)=numerator/deminator;
%             end
%         end
        similarity_Criteria{1,i}=corrcoef(dataSet'); %similarity;
    end
end
end

if strcmp(selection,'item')==1
if iscell(Train_All)==0
    dataSet=Train_All';
    for j=1:size(dataSet,1)
        mean_X=mean(dataSet(j,:));
        for k=1:size(dataSet,1)
            mean_Y=mean(dataSet(k,:));
            [coRatedValue,coRatedItem1,coRatedItem2]=intersect(find(dataSet(j,:)~=0),find(dataSet(k,:)~=0));
            numerator=sum((dataSet(j,coRatedValue)-mean_X).*(dataSet(k,coRatedValue)-mean_Y));
            deminator=sqrt(sum((dataSet(j,coRatedValue)-mean_X).^2))*sqrt(sum((dataSet(k,coRatedValue)-mean_Y).^2));
            similarity(j,k)=numerator/deminator;
        end
    end
    similarity_Criteria=similarity;
else
    for i=1:size(Train_All,2)
        similarity=[];
        dataSet=Train_All{1,i};
%         for j=1:size(dataSet,1)
%             mean_X=mean(dataSet(j,:));
%             for k=1:size(dataSet,1)
%                 mean_Y=mean(dataSet(k,:));
%                 [coRatedValue,coRatedItem1,coRatedItem2]=intersect(find(dataSet(j,:)~=0),find(dataSet(k,:)~=0));
%                 numerator=sum((dataSet(j,coRatedValue)-mean_X).*(dataSet(k,coRatedValue)-mean_Y));
%                 deminator=sqrt(sum((dataSet(j,coRatedValue)-mean_X).^2))*sqrt(sum((dataSet(k,coRatedValue)-mean_Y).^2));
%                 similarity(j,k)=numerator/deminator;
%             end
%         end
        similarity_Criteria{1,i}=corrcoef(dataSet); %similarity;
    end
end
end
end

