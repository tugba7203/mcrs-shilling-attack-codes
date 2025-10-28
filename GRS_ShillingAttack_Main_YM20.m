clear;
close all;
clc;
Data=load('YM20.mat').YM20;
EOverall=Data.O;
EStory=Data.S;
EActing=Data.A;
EDirection=Data.D;
EVisual=Data.V;
[satir, sutun]=size(EOverall);
Overall=EOverall;
Story=EStory;
Acting=EActing;
Direction=EDirection;
Visual=EVisual;
krr=1;

% for i=satir:-1:1
%     if(nnz(EOverall(i,:))<(sutun*0.1))
%         Overall(i,:)=[];
%     end
% end
% for i=satir:-1:1
%     if(nnz(EStory(i,:))<(sutun*0.1))
%         Story(i,:)=[];
%     end
% end
% for i=satir:-1:1
%     if(nnz(EActing(i,:))<(sutun*0.1))
%         Acting(i,:)=[];
%     end
% end
% for i=satir:-1:1
%     if(nnz(EDirection(i,:))<(sutun*0.1))
%         Direction(i,:)=[];
%     end
% end
% for i=satir:-1:1
%     if(nnz(EVisual(i,:))<(sutun*0.1))
%         Visual(i,:)=[];
%     end
% end
[satir, sutun]=size(Overall);
% indices=load('indices_YM20.mat').indices;
indices = crossvalind('Kfold', satir, 10);
% evaluation = evalclusters(TrainO,"kmeans","silhouette","KList",1:20)
k=round(satir*0.05); % küme sayısı
cValue='TR'; % ID, AS, NR, DR, TR
Power_N=50;
powerUser_ratio=0.1; % Attack Size
attack_size=0.1;
metric='MUP_Ratio'; % MUP_Sum, MUP_Nnz, MUP_Th, MUP_Ratio
attackIntent='PUSH';
itemRatio=100;
cluster_size=round(satir*0.01);
en_yakin_komsu=50;
esikDeger=3;
poweruserRatio_homogenity=0.5;
filler_size_ratio=0.01;
% küme sayısının %1-5-10 şeklinde bölelim.
for t=1:10
    %     t=3
    TestO=Overall(indices==t,:);
    TestS=Story(indices==t,:);
    TestA=Acting(indices==t,:);
    TestD=Direction(indices==t,:);
    TestV=Visual(indices==t,:);

    TrainO=Overall(indices~=t,:);
    TrainS=Story(indices~=t,:);
    TrainA=Acting(indices~=t,:);
    TrainD=Direction(indices~=t,:);
    TrainV=Visual(indices~=t,:);
    Train_All={TrainA,TrainS,TrainD,TrainV,TrainO};

    %% Güçlü kullanıcıların belirlenip sisteme sahte profillerin eklenmesi
    [similarityAll] = similarity_measurePCC(Train_All,'user');
    [powerUser_S,powerUser_A,powerUser_D,powerUser_V ,powerUser_O] = find_PowerUsers(similarityAll,Train_All,Power_N,cValue );
    powerUserAll={powerUser_S,powerUser_A,powerUser_D,powerUser_V,powerUser_O};
    [powerUser_dist]=homogenity_powerUser(powerUserAll,similarityAll,cluster_size,poweruserRatio_homogenity,TrainO,'Sim');
    % Power user TrustValue
    [trustValue] = trust_measure(Train_All);
    [powerUser_S_TR,powerUser_A_TR,powerUser_D_TR,powerUser_V_TR ,powerUser_O_TR] = find_PowerUsers(similarityAll,Train_All,Power_N,'TR' );
    powerUserAll_TR={powerUser_S_TR,powerUser_A_TR,powerUser_D_TR,powerUser_V_TR,powerUser_O_TR};
    %% Homojen kümeleme için güçlü kullanıcıların ayrıştırılması
    
    [powerUser_dist_TR]=homogenity_powerUser(powerUserAll_TR,trustValue,cluster_size,poweruserRatio_homogenity,TrainO,'Trust');

    %% *********************  OVERALL ****************************
    % nonZeros=zeros(size(TrainO,1),size(TrainO,2));
    data=TrainO;
    powerData=powerUser_dist{1,5};
    powerData_TR=powerUser_dist_TR{1,5};
    for i=1:size(data,2)
        numberOfNNZ(1,i)=nnz(data(:,i));
    end
    [value_numberOfNNZ indis_numberOfNNZ]=sort(numberOfNNZ,'ascend');
    otherItems=indis_numberOfNNZ(1,1:itemRatio);
    [targetItems] = targetItemSelection_GRS(data,otherItems,metric,attackIntent);
    newData=zeros(size(data,1),size(data,2));
    for c=1:3
        topN=5*c;
        multiple_targetItems=targetItems(1,1:topN);
        selectedItem=[otherItems multiple_targetItems ];
        %         newData=data(:,selectedItem);
        powerUser=powerData; %(1:(size(data,1)*powerUser_ratio));
        powerUser_TR=powerData_TR;
        %         powerUser_data=data(powerUser,selectedItem);
        %         powerUser_data(:,((size(powerUser_data,2)-topN+1):size(powerUser_data,2)))=max(data(:));
        attackPowerUser_data=data(powerUser,:);
        attackPowerUser_data_TR=data(powerUser_TR,:);

        attackPowerUser_data(:,multiple_targetItems)=max(data(:));
        attackPowerUser_data_TR(:,multiple_targetItems)=max(data(:));
        %% Other Attack Models, GSAGen_L Rand ,Avg
        [RA_Data,AA_Data] =GSAGen_L(Train_All,filler_size_ratio,attack_size,multiple_targetItems);
        random_AttackData=RA_Data{1,5};
        average_AttackData=AA_Data{1,5};

        train_AttackData=[data;attackPowerUser_data];
        train_AttackData_TR=[data;attackPowerUser_data_TR];
        [idxPA,CPA] = kmeans(train_AttackData,cluster_size);
        [idxRA,CRA] = kmeans(random_AttackData,cluster_size);
        [idxAA,CAA] = kmeans(average_AttackData,cluster_size);
        [idxPA_TR,CPA_TR] = kmeans(train_AttackData_TR,cluster_size);

        % [standard_data, mu, sigma] = zscore(train_AttackData);     % standardize data so that the mean is 0 and the variance is 1 for each variable
        % [coeff, score, ~]  = pca(standard_data);     % perform PCA
        % new_C = (CPA-mu)./sigma*coeff;     % apply the PCA transformation to the centroid data
        % scatter(score(1:343, 1), score(1:343, 2), [], idxPA(1:343))     % plot 2 principal components of the cluster data (three clusters are shown in different colors)
        % hold on
        % scatter(score(344:377, 1), score(344:377, 2), [], idxPA(344:377),'filled','d','MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor','red')
        % plot(new_C(:, 1), new_C(:, 2), 'kx','MarkerSize',15)


        [Pred_Train_PA] = cf_uygulama(train_AttackData,en_yakin_komsu);
        [Pred_Train_RA] = cf_uygulama(random_AttackData,en_yakin_komsu);
        [Pred_Train_AA] = cf_uygulama(average_AttackData,en_yakin_komsu);
        [Pred_Train_PA_TR] = cf_uygulama(train_AttackData_TR,en_yakin_komsu);
        %          [idxPC1,CPC] = kmeans(Pred_Train,cluster_size);
        Pred_Train_PA((size(Pred_Train_PA,1)-size(powerUser,1)+1):size(Pred_Train_PA,1),multiple_targetItems)=max(data(:));
        Pred_Train_RA((size(Pred_Train_RA,1)-size(powerUser,1)+1):size(Pred_Train_RA,1),multiple_targetItems)=max(data(:));
        Pred_Train_AA((size(Pred_Train_AA,1)-size(powerUser,1)+1):size(Pred_Train_AA,1),multiple_targetItems)=max(data(:));
        Pred_Train_PA_TR((size(Pred_Train_PA_TR,1)-size(powerUser_TR,1)+1):size(Pred_Train_PA_TR,1),multiple_targetItems)=max(data(:));

        PA_uniqueCluster=unique(idxPA(size(train_AttackData,1)-size(powerUser,1):size(train_AttackData,1),:));
        RA_uniqueCluster=unique(idxRA(size(random_AttackData,1)-size(powerUser,1):size(random_AttackData,1),:));
        AA_uniqueCluster=unique(idxAA(size(average_AttackData,1)-size(powerUser,1):size(average_AttackData,1),:));
        PA_uniqueCluster_TR=unique(idxPA_TR(size(train_AttackData_TR,1)-size(powerUser_TR,1):size(train_AttackData_TR,1),:));

        %% -------------------------- PUA MODELİ-----------------------------
        % SIMILARITY
        sayac=1;
        for i=1:cluster_size
%             if find(PA_uniqueCluster==i)
                Group_PA{i}=Pred_Train_PA(idxPA==i,:);
                GrupPA=Group_PA{1,i};
                GroupRatingKMMP_PA(sayac,1:size(GrupPA,2))=max(GrupPA);
                GroupRatingKMLM_PA(sayac,1:size(GrupPA,2))=min(GrupPA);
                [GroupRatingMRP] = mrp_Aggregation(GrupPA);
                GroupRatingKMMRP_PA(sayac,:)=GroupRatingMRP;
                for j=1:size(GrupPA,2)
                    %                 if size(GrupPA,1)<10
                    %                     GroupRatingKMAvg_PA(i,j)=0;
                    %                     GroupRatingKMAU_PA(i,j)=0;
                    %                     GroupRatingKMMUL_PA(i,j)=0;
                    %                     GroupRatingKMSC_PA(i,j)=0;
                    %                     GroupRatingKMAwM_PA(i,j)=0;
                    %                     GroupRatingKMAV_PA(i,j)=0;
                    %                 else
                    GroupRatingKMAvg_PA(sayac,j)=sum(GrupPA(:,j))/nnz(GrupPA(:,j));
                    GroupRatingKMAU_PA(sayac,j)=sum(GrupPA(:,j));
                    GroupRatingKMMUL_PA(sayac,j)=prod(GrupPA(:,j));
                    GroupRatingKMSC_PA(sayac,j)=nnz(GrupPA(:,j));
                    GroupRatingKMAV_PA(sayac,j)=size(GrupPA(find(GrupPA(:,j)>=esikDeger),j),1);
                    GroupRatingKMAwM_PA(sayac,j)=sum(GrupPA(find(GrupPA(:,j)>=esikDeger),j))/size(find(GrupPA(:,j)>=esikDeger),1);
                    %                 end
                end

                [hit_Avg,topn_out_Avg,topn_idx_Avg] = topn_recom(GroupRatingKMAvg_PA,topN,selectedItem);
                [hit_AU,topn_out_AU,topn_idx_AU] = topn_recom(GroupRatingKMAU_PA,topN,selectedItem);
                [hit_MUL,topn_out_MUL,topn_idx_MUL] = topn_recom(GroupRatingKMMUL_PA,topN,selectedItem);
                [hit_SC,topn_out_SC,topn_idx_SC] = topn_recom(GroupRatingKMSC_PA,topN,selectedItem);
                [hit_AwM,topn_out_AwM,topn_idx_AwM] = topn_recom(GroupRatingKMAwM_PA,topN,selectedItem);
                [hit_AV,topn_out_AV,topn_idx_AV] = topn_recom(GroupRatingKMAV_PA,topN,selectedItem);
                [hit_MP,topn_out_MP,topn_idx_MP] = topn_recom(GroupRatingKMMP_PA,topN,selectedItem);
                [hit_LM,topn_out_LM,topn_idx_LM] = topn_recom(GroupRatingKMLM_PA,topN,selectedItem);
                [hit_MRP,topn_out_MRP,topn_idx_MRP] = topn_recom(GroupRatingKMMRP_PA,topN,selectedItem);
                hit_Cluster_PA(:,sayac)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
                sayac=sayac+1;
%             else continue;
%             end
        end

             %% -------------------------- PUA MODELİ-----------------------------
             % ------------------------------------ TRUST VALUE--------
        sayac=1;
        for i=1:cluster_size
            if find(PA_uniqueCluster_TR==i)
                Group_PA_TR{i}=Pred_Train_PA_TR(idxPA_TR==i,:);
                GrupPA_TR=Group_PA_TR{1,i};
                GroupRatingKMMP_PA_TR(sayac,1:size(GrupPA_TR,2))=max(GrupPA_TR);
                GroupRatingKMLM_PA_TR(sayac,1:size(GrupPA_TR,2))=min(GrupPA_TR);
                [GroupRatingMRP_TR] = mrp_Aggregation(GrupPA_TR);
                GroupRatingKMMRP_PA_TR(sayac,:)=GroupRatingMRP_TR;
                for j=1:size(GrupPA_TR,2)
                    %                 if size(GrupPA,1)<10
                    %                     GroupRatingKMAvg_PA(i,j)=0;
                    %                     GroupRatingKMAU_PA(i,j)=0;
                    %                     GroupRatingKMMUL_PA(i,j)=0;
                    %                     GroupRatingKMSC_PA(i,j)=0;
                    %                     GroupRatingKMAwM_PA(i,j)=0;
                    %                     GroupRatingKMAV_PA(i,j)=0;
                    %                 else
                    GroupRatingKMAvg_PA_TR(sayac,j)=sum(GrupPA_TR(:,j))/nnz(GrupPA_TR(:,j));
                    GroupRatingKMAU_PA_TR(sayac,j)=sum(GrupPA_TR(:,j));
                    GroupRatingKMMUL_PA_TR(sayac,j)=prod(GrupPA_TR(:,j));
                    GroupRatingKMSC_PA_TR(sayac,j)=nnz(GrupPA_TR(:,j));
                    GroupRatingKMAV_PA_TR(sayac,j)=size(GrupPA_TR(find(GrupPA_TR(:,j)>=esikDeger),j),1);
                    GroupRatingKMAwM_PA_TR(sayac,j)=sum(GrupPA_TR(find(GrupPA_TR(:,j)>=esikDeger),j))/size(find(GrupPA_TR(:,j)>=esikDeger),1);
                    %                 end
                end

                [hit_Avg,topn_out_Avg,topn_idx_Avg] = topn_recom(GroupRatingKMAvg_PA_TR,topN,selectedItem);
                [hit_AU,topn_out_AU,topn_idx_AU] = topn_recom(GroupRatingKMAU_PA_TR,topN,selectedItem);
                [hit_MUL,topn_out_MUL,topn_idx_MUL] = topn_recom(GroupRatingKMMUL_PA_TR,topN,selectedItem);
                [hit_SC,topn_out_SC,topn_idx_SC] = topn_recom(GroupRatingKMSC_PA_TR,topN,selectedItem);
                [hit_AwM,topn_out_AwM,topn_idx_AwM] = topn_recom(GroupRatingKMAwM_PA_TR,topN,selectedItem);
                [hit_AV,topn_out_AV,topn_idx_AV] = topn_recom(GroupRatingKMAV_PA_TR,topN,selectedItem);
                [hit_MP,topn_out_MP,topn_idx_MP] = topn_recom(GroupRatingKMMP_PA_TR,topN,selectedItem);
                [hit_LM,topn_out_LM,topn_idx_LM] = topn_recom(GroupRatingKMLM_PA_TR,topN,selectedItem);
                [hit_MRP,topn_out_MRP,topn_idx_MRP] = topn_recom(GroupRatingKMMRP_PA_TR,topN,selectedItem);
                hit_Cluster_PA_TR(:,sayac)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
                sayac=sayac+1;
            else continue;
            end
        end

        %% ------------------------ RANDOM ATTACK----------------------------
        sayac=1;
        for i=1:cluster_size
            if find(RA_uniqueCluster==i)
                Group_RA{i}=Pred_Train_RA(idxRA==i,:);
                GrupRA=Group_RA{1,i};
                GroupRatingKMMP_RA(sayac,1:size(GrupRA,2))=max(GrupRA);
                GroupRatingKMLM_RA(sayac,1:size(GrupRA,2))=min(GrupRA);
                [GroupRatingMRP] = mrp_Aggregation(GrupRA);
                GroupRatingKMMRP_RA(sayac,:)=GroupRatingMRP;
                for j=1:size(GrupRA,2)
                    %                 if size(GrupRA,1)<10
                    %                     GroupRatingKMAvg_RA(i,j)=0;
                    %                     GroupRatingKMAU_RA(i,j)=0;
                    %                     GroupRatingKMMUL_RA(i,j)=0;
                    %                     GroupRatingKMSC_RA(i,j)=0;
                    %                     GroupRatingKMAwM_RA(i,j)=0;
                    %                     GroupRatingKMAV_RA(i,j)=0;
                    %                 else
                    GroupRatingKMAvg_RA(sayac,j)=sum(GrupRA(:,j))/nnz(GrupRA(:,j));
                    GroupRatingKMAU_RA(sayac,j)=sum(GrupRA(:,j));
                    GroupRatingKMMUL_RA(sayac,j)=prod(GrupRA(:,j));
                    GroupRatingKMSC_RA(sayac,j)=nnz(GrupRA(:,j));
                    GroupRatingKMAV_RA(sayac,j)=size(GrupRA(find(GrupRA(:,j)>=esikDeger),j),1);
                    GroupRatingKMAwM_RA(sayac,j)=sum(GrupRA(find(GrupRA(:,j)>=esikDeger),j))/size(find(GrupRA(:,j)>=esikDeger),1);
                    %                 end
                end
                [hit_Avg,topn_out_Avg,topn_idx_Avg] = topn_recom(GroupRatingKMAvg_RA,topN,selectedItem);
                [hit_AU,topn_out_AU,topn_idx_AU] = topn_recom(GroupRatingKMAU_RA,topN,selectedItem);
                [hit_MUL,topn_out_MUL,topn_idx_MUL] = topn_recom(GroupRatingKMMUL_RA,topN,selectedItem);
                [hit_SC,topn_out_SC,topn_idx_SC] = topn_recom(GroupRatingKMSC_RA,topN,selectedItem);
                [hit_AwM,topn_out_AwM,topn_idx_AwM] = topn_recom(GroupRatingKMAwM_RA,topN,selectedItem);
                [hit_AV,topn_out_AV,topn_idx_AV] = topn_recom(GroupRatingKMAV_RA,topN,selectedItem);
                [hit_MP,topn_out_MP,topn_idx_MP] = topn_recom(GroupRatingKMMP_RA,topN,selectedItem);
                [hit_LM,topn_out_LM,topn_idx_LM] = topn_recom(GroupRatingKMLM_RA,topN,selectedItem);
                [hit_MRP,topn_out_MRP,topn_idx_MRP] = topn_recom(GroupRatingKMMRP_RA,topN,selectedItem);
                hit_Cluster_RA(:,sayac)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
                sayac=sayac+1;
            else
                continue;
            end
        end
        sayac=1;
        %% ------------------------------ AVERAGE ATTACK ---------------------------
        for i=1:cluster_size
            if find(AA_uniqueCluster==i)
                Group_AA{i}=Pred_Train_AA(idxAA==i,:);
                GrupAA=Group_AA{1,i};
                GroupRatingKMMP_AA(sayac,1:size(GrupAA,2))=max(GrupAA);
                GroupRatingKMLM_AA(sayac,1:size(GrupAA,2))=min(GrupAA);
                [GroupRatingMRP] = mrp_Aggregation(GrupAA);
                GroupRatingKMMRP_AA(sayac,:)=GroupRatingMRP;
                for j=1:size(GrupAA,2)
                    %                 if size(GrupAA,1)<10
                    %                     GroupRatingKMAvg_AA(i,j)=0;
                    %                     GroupRatingKMAU_AA(i,j)=0;
                    %                     GroupRatingKMMUL_AA(i,j)=0;
                    %                     GroupRatingKMSC_AA(i,j)=0;
                    %                     GroupRatingKMAwM_AA(i,j)=0;
                    %                     GroupRatingKMAV_AA(i,j)=0;
                    %                 else
                    GroupRatingKMAvg_AA(sayac,j)=sum(GrupAA(:,j))/nnz(GrupAA(:,j));
                    GroupRatingKMAU_AA(sayac,j)=sum(GrupAA(:,j));
                    GroupRatingKMMUL_AA(sayac,j)=prod(GrupAA(:,j));
                    GroupRatingKMSC_AA(sayac,j)=nnz(GrupAA(:,j));
                    GroupRatingKMAV_AA(sayac,j)=size(GrupAA(find(GrupAA(:,j)>=esikDeger),j),1);
                    GroupRatingKMAwM_AA(sayac,j)=sum(GrupAA(find(GrupAA(:,j)>=esikDeger),j))/size(find(GrupAA(:,j)>=esikDeger),1);
                    %                 end
                end

                [hit_Avg,topn_out_Avg,topn_idx_Avg] = topn_recom(GroupRatingKMAvg_AA,topN,selectedItem);
                [hit_AU,topn_out_AU,topn_idx_AU] = topn_recom(GroupRatingKMAU_AA,topN,selectedItem);
                [hit_MUL,topn_out_MUL,topn_idx_MUL] = topn_recom(GroupRatingKMMUL_AA,topN,selectedItem);
                [hit_SC,topn_out_SC,topn_idx_SC] = topn_recom(GroupRatingKMSC_AA,topN,selectedItem);
                [hit_AwM,topn_out_AwM,topn_idx_AwM] = topn_recom(GroupRatingKMAwM_AA,topN,selectedItem);
                [hit_AV,topn_out_AV,topn_idx_AV] = topn_recom(GroupRatingKMAV_AA,topN,selectedItem);
                [hit_MP,topn_out_MP,topn_idx_MP] = topn_recom(GroupRatingKMMP_AA,topN,selectedItem);
                [hit_LM,topn_out_LM,topn_idx_LM] = topn_recom(GroupRatingKMLM_AA,topN,selectedItem);
                [hit_MRP,topn_out_MRP,topn_idx_MRP] = topn_recom(GroupRatingKMMRP_AA,topN,selectedItem);
                hit_Cluster_AA(:,sayac)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
                sayac=sayac+1;
            else
                continue;
            end


        end
        overall_hit_PA{t,c}=hit_Cluster_PA;
        mean_hit_PA{t,c}=mean(hit_Cluster_PA);

        overall_hit_RA{t,c}=hit_Cluster_RA;
        mean_hit_RA{t,c}=mean(hit_Cluster_RA);

        overall_hit_AA{t,c}=hit_Cluster_AA;
        mean_hit_AA{t,c}=mean(hit_Cluster_AA);

        overall_hit_PA_TR{t,c}=hit_Cluster_PA_TR;
        mean_hit_PA_TR{t,c}=mean(hit_Cluster_PA_TR);

    end
    % ******************** OVERALL TABANLI **************************************************************************
    %************************SONU***********************************************************************************
    %***************************************************************************************************************

    %% ******************** KRİTER TABANLI ********************************
    % **********************************************************************
    % ************************************************************************
    for kriter=1:(size(Train_All,2)-1)
        data=Train_All{1,kriter};
        powerData=powerUser_dist{1,kriter};
        for i=1:size(data,2)
            numberOfNNZ(1,i)=nnz(data(:,i));
        end
        [value_numberOfNNZ indis_numberOfNNZ]=sort(numberOfNNZ,'ascend');
        otherItems=indis_numberOfNNZ(1,1:itemRatio);
        [targetItems] = targetItemSelection_GRS(data,otherItems,metric,attackIntent);
        newData=zeros(size(data,1),size(data,2));
        for c=1:3
            topN=5*c;
            multiple_targetItems=targetItems(1,1:topN);
            selectedItem=[otherItems multiple_targetItems ];
            %         newData=data(:,selectedItem);
            powerUser=powerData(1:(size(data,1)*powerUser_ratio));
            %         powerUser_data=data(powerUser,selectedItem);
            %         powerUser_data(:,((size(powerUser_data,2)-topN+1):size(powerUser_data,2)))=max(data(:));
            attackPowerUser_data=data(powerUser,:);
            attackPowerUser_data(:,multiple_targetItems)=max(data(:));
            %% Other Attack Models, GSAGen_L Rand ,Avg
            [RA_Data,AA_Data] =GSAGen_L(Train_All,filler_size_ratio,attack_size,multiple_targetItems);
            random_AttackData=RA_Data{1,5};
            average_AttackData=AA_Data{1,5};

            train_AttackData=[data;attackPowerUser_data];
            [idxPA,CPA] = kmeans(train_AttackData,cluster_size);
            [idxRA,CRA] = kmeans(random_AttackData,cluster_size);
            [idxAA,CAA] = kmeans(average_AttackData,cluster_size);

            % [standard_data, mu, sigma] = zscore(train_AttackData);     % standardize data so that the mean is 0 and the variance is 1 for each variable
            % [coeff, score, ~]  = pca(standard_data);     % perform PCA
            % new_C = (CPA-mu)./sigma*coeff;     % apply the PCA transformation to the centroid data
            % scatter(score(1:343, 1), score(1:343, 2), [], idxPA(1:343))     % plot 2 principal components of the cluster data (three clusters are shown in different colors)
            % hold on
            % scatter(score(344:377, 1), score(344:377, 2), [], idxPA(344:377),'filled','d','MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor','red')
            % plot(new_C(:, 1), new_C(:, 2), 'kx','MarkerSize',15)


            [Pred_Train_PA] = cf_uygulama(train_AttackData,en_yakin_komsu);
            [Pred_Train_RA] = cf_uygulama(random_AttackData,en_yakin_komsu);
            [Pred_Train_AA] = cf_uygulama(average_AttackData,en_yakin_komsu);
            %          [idxPC1,CPC] = kmeans(Pred_Train,cluster_size);
            Pred_Train_PA((size(Pred_Train_PA,1)-size(powerUser,1)+1):size(Pred_Train_PA,1),multiple_targetItems)=max(data(:));
            Pred_Train_RA((size(Pred_Train_RA,1)-size(powerUser,1)+1):size(Pred_Train_RA,1),multiple_targetItems)=max(data(:));
            Pred_Train_AA((size(Pred_Train_AA,1)-size(powerUser,1)+1):size(Pred_Train_AA,1),multiple_targetItems)=max(data(:));

            PA_uniqueCluster=unique(idxPA(size(train_AttackData,1)-size(powerUser,1):size(train_AttackData,1),:));
            RA_uniqueCluster=unique(idxRA(size(random_AttackData,1)-size(powerUser,1):size(random_AttackData,1),:));
            AA_uniqueCluster=unique(idxAA(size(average_AttackData,1)-size(powerUser,1):size(average_AttackData,1),:));

            %% -------------------------- PUA MODELİ-----------------------------
%             sayac=1;
%             for i=1:cluster_size
%                 if find(PA_uniqueCluster==i)
%                     Group_PA{i}=Pred_Train_PA(idxPA==i,:);
%                     GrupPA=Group_PA{1,i};
%                     GroupRatingKMMP_PA(sayac,1:size(GrupPA,2))=max(GrupPA);
%                     GroupRatingKMLM_PA(sayac,1:size(GrupPA,2))=min(GrupPA);
%                     [GroupRatingMRP] = mrp_Aggregation(GrupPA);
%                     GroupRatingKMMRP_PA(sayac,:)=GroupRatingMRP;
%                     for j=1:size(GrupPA,2)
%                         %                 if size(GrupPA,1)<10
%                         %                     GroupRatingKMAvg_PA(i,j)=0;
%                         %                     GroupRatingKMAU_PA(i,j)=0;
%                         %                     GroupRatingKMMUL_PA(i,j)=0;
%                         %                     GroupRatingKMSC_PA(i,j)=0;
%                         %                     GroupRatingKMAwM_PA(i,j)=0;
%                         %                     GroupRatingKMAV_PA(i,j)=0;
%                         %                 else
%                         GroupRatingKMAvg_PA(sayac,j)=sum(GrupPA(:,j))/nnz(GrupPA(:,j));
%                         GroupRatingKMAU_PA(sayac,j)=sum(GrupPA(:,j));
%                         GroupRatingKMMUL_PA(sayac,j)=prod(GrupPA(:,j));
%                         GroupRatingKMSC_PA(sayac,j)=nnz(GrupPA(:,j));
%                         GroupRatingKMAV_PA(sayac,j)=size(GrupPA(find(GrupPA(:,j)>=esikDeger),j),1);
%                         GroupRatingKMAwM_PA(sayac,j)=sum(GrupPA(find(GrupPA(:,j)>=esikDeger),j))/size(find(GrupPA(:,j)>=esikDeger),1);
%                         %                 end
%                     end
% 
%                     [hit_Avg,topn_out_Avg,topn_idx_Avg] = topn_recom(GroupRatingKMAvg_PA,topN,selectedItem);
%                     [hit_AU,topn_out_AU,topn_idx_AU] = topn_recom(GroupRatingKMAU_PA,topN,selectedItem);
%                     [hit_MUL,topn_out_MUL,topn_idx_MUL] = topn_recom(GroupRatingKMMUL_PA,topN,selectedItem);
%                     [hit_SC,topn_out_SC,topn_idx_SC] = topn_recom(GroupRatingKMSC_PA,topN,selectedItem);
%                     [hit_AwM,topn_out_AwM,topn_idx_AwM] = topn_recom(GroupRatingKMAwM_PA,topN,selectedItem);
%                     [hit_AV,topn_out_AV,topn_idx_AV] = topn_recom(GroupRatingKMAV_PA,topN,selectedItem);
%                     [hit_MP,topn_out_MP,topn_idx_MP] = topn_recom(GroupRatingKMMP_PA,topN,selectedItem);
%                     [hit_LM,topn_out_LM,topn_idx_LM] = topn_recom(GroupRatingKMLM_PA,topN,selectedItem);
%                     [hit_MRP,topn_out_MRP,topn_idx_MRP] = topn_recom(GroupRatingKMMRP_PA,topN,selectedItem);
%                     hit_Cluster_PA(:,sayac)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
%                     sayac=sayac+1;
%                 else continue;
%                 end
%             end

            %% ------------------------ RANDOM ATTACK----------------------------
            sayac=1;
            for i=1:cluster_size
                if find(RA_uniqueCluster==i)
                    Group_RA{i}=Pred_Train_RA(idxRA==i,:);
                    GrupRA=Group_RA{1,i};
                    GroupRatingKMMP_RA(sayac,1:size(GrupRA,2))=max(GrupRA);
                    GroupRatingKMLM_RA(sayac,1:size(GrupRA,2))=min(GrupRA);
                    [GroupRatingMRP] = mrp_Aggregation(GrupRA);
                    GroupRatingKMMRP_RA(sayac,:)=GroupRatingMRP;
                    for j=1:size(GrupRA,2)
                        %                 if size(GrupRA,1)<10
                        %                     GroupRatingKMAvg_RA(i,j)=0;
                        %                     GroupRatingKMAU_RA(i,j)=0;
                        %                     GroupRatingKMMUL_RA(i,j)=0;
                        %                     GroupRatingKMSC_RA(i,j)=0;
                        %                     GroupRatingKMAwM_RA(i,j)=0;
                        %                     GroupRatingKMAV_RA(i,j)=0;
                        %                 else
                        GroupRatingKMAvg_RA(sayac,j)=sum(GrupRA(:,j))/nnz(GrupRA(:,j));
                        GroupRatingKMAU_RA(sayac,j)=sum(GrupRA(:,j));
                        GroupRatingKMMUL_RA(sayac,j)=prod(GrupRA(:,j));
                        GroupRatingKMSC_RA(sayac,j)=nnz(GrupRA(:,j));
                        GroupRatingKMAV_RA(sayac,j)=size(GrupRA(find(GrupRA(:,j)>=esikDeger),j),1);
                        GroupRatingKMAwM_RA(sayac,j)=sum(GrupRA(find(GrupRA(:,j)>=esikDeger),j))/size(find(GrupRA(:,j)>=esikDeger),1);
                        %                 end
                    end
                    [hit_Avg,topn_out_Avg,topn_idx_Avg] = topn_recom(GroupRatingKMAvg_RA,topN,selectedItem);
                    [hit_AU,topn_out_AU,topn_idx_AU] = topn_recom(GroupRatingKMAU_RA,topN,selectedItem);
                    [hit_MUL,topn_out_MUL,topn_idx_MUL] = topn_recom(GroupRatingKMMUL_RA,topN,selectedItem);
                    [hit_SC,topn_out_SC,topn_idx_SC] = topn_recom(GroupRatingKMSC_RA,topN,selectedItem);
                    [hit_AwM,topn_out_AwM,topn_idx_AwM] = topn_recom(GroupRatingKMAwM_RA,topN,selectedItem);
                    [hit_AV,topn_out_AV,topn_idx_AV] = topn_recom(GroupRatingKMAV_RA,topN,selectedItem);
                    [hit_MP,topn_out_MP,topn_idx_MP] = topn_recom(GroupRatingKMMP_RA,topN,selectedItem);
                    [hit_LM,topn_out_LM,topn_idx_LM] = topn_recom(GroupRatingKMLM_RA,topN,selectedItem);
                    [hit_MRP,topn_out_MRP,topn_idx_MRP] = topn_recom(GroupRatingKMMRP_RA,topN,selectedItem);
                    hit_Cluster_RA(:,sayac)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
                    sayac=sayac+1;
                else
                    continue;
                end
            end
            sayac=1;
            %% ------------------------------ AVERAGE ATTACK ---------------------------
            for i=1:cluster_size
                if find(AA_uniqueCluster==i)
                    Group_AA{i}=Pred_Train_AA(idxAA==i,:);
                    GrupAA=Group_AA{1,i};
                    GroupRatingKMMP_AA(sayac,1:size(GrupAA,2))=max(GrupAA);
                    GroupRatingKMLM_AA(sayac,1:size(GrupAA,2))=min(GrupAA);
                    [GroupRatingMRP] = mrp_Aggregation(GrupAA);
                    GroupRatingKMMRP_AA(sayac,:)=GroupRatingMRP;
                    for j=1:size(GrupAA,2)
                        %                 if size(GrupAA,1)<10
                        %                     GroupRatingKMAvg_AA(i,j)=0;
                        %                     GroupRatingKMAU_AA(i,j)=0;
                        %                     GroupRatingKMMUL_AA(i,j)=0;
                        %                     GroupRatingKMSC_AA(i,j)=0;
                        %                     GroupRatingKMAwM_AA(i,j)=0;
                        %                     GroupRatingKMAV_AA(i,j)=0;
                        %                 else
                        GroupRatingKMAvg_AA(sayac,j)=sum(GrupAA(:,j))/nnz(GrupAA(:,j));
                        GroupRatingKMAU_AA(sayac,j)=sum(GrupAA(:,j));
                        GroupRatingKMMUL_AA(sayac,j)=prod(GrupAA(:,j));
                        GroupRatingKMSC_AA(sayac,j)=nnz(GrupAA(:,j));
                        GroupRatingKMAV_AA(sayac,j)=size(GrupAA(find(GrupAA(:,j)>=esikDeger),j),1);
                        GroupRatingKMAwM_AA(sayac,j)=sum(GrupAA(find(GrupAA(:,j)>=esikDeger),j))/size(find(GrupAA(:,j)>=esikDeger),1);
                        %                 end
                    end

                    [hit_Avg,topn_out_Avg,topn_idx_Avg] = topn_recom(GroupRatingKMAvg_AA,topN,selectedItem);
                    [hit_AU,topn_out_AU,topn_idx_AU] = topn_recom(GroupRatingKMAU_AA,topN,selectedItem);
                    [hit_MUL,topn_out_MUL,topn_idx_MUL] = topn_recom(GroupRatingKMMUL_AA,topN,selectedItem);
                    [hit_SC,topn_out_SC,topn_idx_SC] = topn_recom(GroupRatingKMSC_AA,topN,selectedItem);
                    [hit_AwM,topn_out_AwM,topn_idx_AwM] = topn_recom(GroupRatingKMAwM_AA,topN,selectedItem);
                    [hit_AV,topn_out_AV,topn_idx_AV] = topn_recom(GroupRatingKMAV_AA,topN,selectedItem);
                    [hit_MP,topn_out_MP,topn_idx_MP] = topn_recom(GroupRatingKMMP_AA,topN,selectedItem);
                    [hit_LM,topn_out_LM,topn_idx_LM] = topn_recom(GroupRatingKMLM_AA,topN,selectedItem);
                    [hit_MRP,topn_out_MRP,topn_idx_MRP] = topn_recom(GroupRatingKMMRP_AA,topN,selectedItem);
                    hit_Cluster_AA(:,sayac)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
                    sayac=sayac+1;
                else
                    continue;
                end
            end
            overall_hit_PA_kriter{t,c}=hit_Cluster_PA;
            mean_hit_PA_kriter{t,c}=mean(hit_Cluster_PA);

            overall_hit_RA_kriter{t,c}=hit_Cluster_RA;
            mean_hit_RA_kriter{t,c}=mean(hit_Cluster_RA);

            overall_hit_AA_kriter{t,c}=hit_Cluster_AA;
            mean_hit_AA_kriter{t,c}=mean(hit_Cluster_AA);

        end
    end




end % iterasyon



