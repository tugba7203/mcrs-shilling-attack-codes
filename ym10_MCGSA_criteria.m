clear;
close all;
clc;
Data=load('YM10.mat').YM10;
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
cValue='ID'; % ID, AS, NR, DR, TR
Power_N=50;
powerUser_ratio=0.1; % Attack Size
en_yakin_komsu=50;
poweruserRatio_homogenity=0.5;
% Parametrik değerlerin eldesi
attackIntent='PUSH';
itemRatio=100;
esikDeger=3;
attack_size=0.1;
clusterBoyut=[0.01,0.05,0.1];
metricName={'MUP_Th','MUP_Ratio','MUP_Nnz','MUP_Sum'};

% cluster_size=round(satir*0.01);
filler_size_ratio=0.05    ;
% metric='MUP_Ratio'; % MUP_Sum, MUP_Nnz, MUP_Th,UsR, MUP_Ratio,dr
% küme sayısının %1-5-10 şeklinde bölelim.
for mN=1:4
    metric=cell2mat(metricName(1,mN));
    for cB=1:3
        cluster_size=round(satir*clusterBoyut(1,cB));
        for t=1:1
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
            Train_All={TrainS,TrainA,TrainD,TrainV,TrainO};

            %% Güçlü kullanıcıların belirlenip sisteme sahte profillerin eklenmesi
            [similarityAll] = similarity_measurePCC(Train_All,'user');
            [similarityAll_item] = similarity_measurePCC(Train_All,'item');
            [powerUser_S,powerUser_A,powerUser_D,powerUser_V ,powerUser_O] = find_PowerUsers(similarityAll,Train_All,Power_N,cValue );
            powerUserAll={powerUser_S,powerUser_A,powerUser_D,powerUser_V,powerUser_O};
            [powerUser_dist]=homogenity_powerUser(powerUserAll,similarityAll,cluster_size,poweruserRatio_homogenity,TrainO,'Sim');
            % Power user TrustValue
            [trustValue] = trust_measure(Train_All);
            [powerUser_S_TR,powerUser_A_TR,powerUser_D_TR,powerUser_V_TR ,powerUser_O_TR] = find_PowerUsers(similarityAll,Train_All,Power_N,'TR' );
            powerUserAll_TR={powerUser_S_TR,powerUser_A_TR,powerUser_D_TR,powerUser_V_TR,powerUser_O_TR};
            %% Homojen kümeleme için güçlü kullanıcıların ayrıştırılması

%             [powerUser_dist_TR]=homogenity_powerUser(powerUserAll_TR,trustValue,cluster_size,poweruserRatio_homogenity,TrainO,'Trust');

            %% *********************  OVERALL ****************************
            % nonZeros=zeros(size(TrainO,1),size(TrainO,2));
            for kriteria=1:4
                data=Train_All{1,kriteria};
                powerData=powerUserAll{1,kriteria};
                powerData_TR=powerUserAll_TR{1,kriteria};
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
%                     powerUser_TR=powerData_TR;
                    %         powerUser_data=data(powerUser,selectedItem);
                    %         powerUser_data(:,((size(powerUser_data,2)-topN+1):size(powerUser_data,2)))=max(data(:));
%                     attackPowerUser_data=data(powerUser,:);
%                     attackPowerUser_data_TR=data(powerUser_TR,:);

%                     attackPowerUser_data(:,multiple_targetItems)=max(data(:));
%                     attackPowerUser_data_TR(:,multiple_targetItems)=max(data(:));
                    %% Other Attack Models, GSAGen_L Rand ,Avg
                    [RA_Data,AA_Data] =GSAGen_L(Train_All,filler_size_ratio,attack_size,multiple_targetItems);
                    random_AttackData=RA_Data{1,5};
                    average_AttackData=AA_Data{1,5};

%                     train_AttackData=[data;attackPowerUser_data];
%                     train_AttackData_TR=[data;attackPowerUser_data_TR];
%                     [idxPA,CPA] = kmeans(train_AttackData,cluster_size);
                    [idxRA,CRA] = kmeans(random_AttackData,cluster_size);
                    [idxAA,CAA] = kmeans(average_AttackData,cluster_size);
%                     [idxPA_TR,CPA_TR] = kmeans(train_AttackData_TR,cluster_size);

%                     [standard_data, mu, sigma] = zscore(train_AttackData);     % standardize data so that the mean is 0 and the variance is 1 for each variable
%                     [coeff, score, ~]  = pca(standard_data);     % perform PCA
%                     new_C = (CPA-mu)./sigma*coeff;     % apply the PCA transformation to the centroid data
% %                     scatter(score(1:343, 1), score(1:343, 2), [], idxPA(1:343))     % plot 2 principal components of the cluster data (three clusters are shown in different colors)
%                     hold on
%                     scatter(score(344:377, 1), score(344:377, 2), [], idxPA(344:377),'filled','d','MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor','red')
%                     plot(new_C(:, 1), new_C(:, 2), 'kx','MarkerSize',15)


%                     [Pred_Train_PA] = cf_uygulama(train_AttackData,en_yakin_komsu);
                    [Pred_Train_RA] = cf_uygulama(random_AttackData,en_yakin_komsu);
                    [Pred_Train_AA] = cf_uygulama(average_AttackData,en_yakin_komsu);
%                     [Pred_Train_PA_TR] = cf_uygulama(train_AttackData_TR,en_yakin_komsu);

%                     Pred_Train_PA((size(Pred_Train_PA,1)-size(powerUser,1)+1):size(Pred_Train_PA,1),multiple_targetItems)=max(data(:));
                    Pred_Train_RA((size(Pred_Train_RA,1)-size(powerUser,1)+1):size(Pred_Train_RA,1),multiple_targetItems)=max(data(:));
                    Pred_Train_AA((size(Pred_Train_AA,1)-size(powerUser,1)+1):size(Pred_Train_AA,1),multiple_targetItems)=max(data(:));
%                     Pred_Train_PA_TR((size(Pred_Train_PA_TR,1)-size(powerUser_TR,1)+1):size(Pred_Train_PA_TR,1),multiple_targetItems)=max(data(:));

%                     PA_uniqueCluster=unique(idxPA(size(train_AttackData,1)-size(powerUser,1):size(train_AttackData,1),:));
                    RA_uniqueCluster=unique(idxRA(size(random_AttackData,1)-size(powerUser,1):size(random_AttackData,1),:));
                    AA_uniqueCluster=unique(idxAA(size(average_AttackData,1)-size(powerUser,1):size(average_AttackData,1),:));
%                     PA_uniqueCluster_TR=unique(idxPA_TR(size(train_AttackData_TR,1)-size(powerUser_TR,1):size(train_AttackData_TR,1),:));



                    %% ------------------------ RANDOM ATTACK----------------------------
                    sayac=1;
                    for i=1:cluster_size
                        %             if find(RA_uniqueCluster==i)
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
                        [hit_Avg,topn_out_Avg,topn_idx_Avg,WRV_Avg] = topn_recom(GroupRatingKMAvg_RA,topN,selectedItem);
                        [ novelty_AVG,diversity_AVG ] = topn_Serendipity( topn_idx_Avg,similarityAll_item,Train_All,topN ,topn_idx_Avg);

                        [hit_AU,topn_out_AU,topn_idx_AU,WRV_AU] = topn_recom(GroupRatingKMAU_RA,topN,selectedItem);
                        [ novelty_AU,diversity_AU ] = topn_Serendipity( topn_idx_AU,similarityAll_item,Train_All,topN ,topn_idx_AU);

                        [hit_MUL,topn_out_MUL,topn_idx_MUL,WRV_MUL] = topn_recom(GroupRatingKMMUL_RA,topN,selectedItem);
                        [ novelty_MUL,diversity_MUL ] = topn_Serendipity( topn_idx_MUL,similarityAll_item,Train_All,topN ,topn_idx_MUL);

                        [hit_SC,topn_out_SC,topn_idx_SC,WRV_SC] = topn_recom(GroupRatingKMSC_RA,topN,selectedItem);
                        [ novelty_SC,diversity_SC] = topn_Serendipity( topn_idx_SC,similarityAll_item,Train_All,topN ,topn_idx_SC);

                        [hit_AwM,topn_out_AwM,topn_idx_AwM,WRV_AwM] = topn_recom(GroupRatingKMAwM_RA,topN,selectedItem);
                        [ novelty_AWM,diversity_AWM ] = topn_Serendipity( topn_idx_AwM,similarityAll_item,Train_All,topN ,topn_idx_AwM);

                        [hit_AV,topn_out_AV,topn_idx_AV,WRV_AV] = topn_recom(GroupRatingKMAV_RA,topN,selectedItem);
                        [ novelty_AV,diversity_AV] = topn_Serendipity( topn_idx_AV,similarityAll_item,Train_All,topN ,topn_idx_AV);

                        [hit_MP,topn_out_MP,topn_idx_MP,WRV_MP] = topn_recom(GroupRatingKMMP_RA,topN,selectedItem);
                        [ novelty_MP,diversity_MP ] = topn_Serendipity( topn_idx_MP,similarityAll_item,Train_All,topN ,topn_idx_MP);

                        [hit_LM,topn_out_LM,topn_idx_LM,WRV_LM] = topn_recom(GroupRatingKMLM_RA,topN,selectedItem);
                        [ novelty_LM,diversity_LM ] = topn_Serendipity( topn_idx_LM,similarityAll_item,Train_All,topN ,topn_idx_LM);

                        [hit_MRP,topn_out_MRP,topn_idx_MRP,WRV_MRP] = topn_recom(GroupRatingKMMRP_RA,topN,selectedItem);
                        [ novelty_MRP,diversity_MRP ] = topn_Serendipity( topn_idx_MRP,similarityAll_item,Train_All,topN ,topn_idx_MRP);

                        hit_Cluster_RA(:,sayac)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
                        novelty_Cluster_RA(:,sayac)=[novelty_AVG;novelty_AU;novelty_MUL;novelty_SC;novelty_AWM;novelty_AV;novelty_MP;novelty_LM;novelty_MRP];
                        diversity_Cluster_RA(:,sayac)=[diversity_AVG;diversity_AU;diversity_MUL;diversity_SC;diversity_AWM;diversity_AV;diversity_MP;diversity_LM;diversity_MRP];
                        WRV_Cluster_RA(:,sayac)=[WRV_Avg;WRV_AU;WRV_MUL;WRV_SC;WRV_AwM;WRV_AV;WRV_MP;WRV_LM;WRV_MRP];
                        sayac=sayac+1;
                        %             else
                        %                 continue;
                        %             end
                    end
                    sayac2=1;
                    %% ------------------------------ AVERAGE ATTACK ---------------------------
                    for i=1:cluster_size
                        %             if find(AA_uniqueCluster==i)
                        Group_AA{i}=Pred_Train_AA(idxAA==i,:);
                        GrupAA=Group_AA{1,i};
                        GroupRatingKMMP_AA(sayac2,1:size(GrupAA,2))=max(GrupAA);
                        GroupRatingKMLM_AA(sayac2,1:size(GrupAA,2))=min(GrupAA);
                        [GroupRatingMRP] = mrp_Aggregation(GrupAA);
                        GroupRatingKMMRP_AA(sayac2,:)=GroupRatingMRP;
                        for j=1:size(GrupAA,2)
                            %                 if size(GrupAA,1)<10
                            %                     GroupRatingKMAvg_AA(i,j)=0;
                            %                     GroupRatingKMAU_AA(i,j)=0;
                            %                     GroupRatingKMMUL_AA(i,j)=0;
                            %                     GroupRatingKMSC_AA(i,j)=0;
                            %                     GroupRatingKMAwM_AA(i,j)=0;
                            %                     GroupRatingKMAV_AA(i,j)=0;
                            %                 else
                            GroupRatingKMAvg_AA(sayac2,j)=sum(GrupAA(:,j))/nnz(GrupAA(:,j));
                            GroupRatingKMAU_AA(sayac2,j)=sum(GrupAA(:,j));
                            GroupRatingKMMUL_AA(sayac2,j)=prod(GrupAA(:,j));
                            GroupRatingKMSC_AA(sayac2,j)=nnz(GrupAA(:,j));
                            GroupRatingKMAV_AA(sayac2,j)=size(GrupAA(find(GrupAA(:,j)>=esikDeger),j),1);
                            GroupRatingKMAwM_AA(sayac2,j)=sum(GrupAA(find(GrupAA(:,j)>=esikDeger),j))/size(find(GrupAA(:,j)>=esikDeger),1);
                            %                 end
                        end

                        [hit_Avg,topn_out_Avg,topn_idx_Avg,WRV_Avg] = topn_recom(GroupRatingKMAvg_AA,topN,selectedItem);
                        [hit_AU,topn_out_AU,topn_idx_AU,WRV_AU] = topn_recom(GroupRatingKMAU_AA,topN,selectedItem);
                        [hit_MUL,topn_out_MUL,topn_idx_MUL,WRV_MUL] = topn_recom(GroupRatingKMMUL_AA,topN,selectedItem);
                        [hit_SC,topn_out_SC,topn_idx_SC,WRV_SC] = topn_recom(GroupRatingKMSC_AA,topN,selectedItem);
                        [hit_AwM,topn_out_AwM,topn_idx_AwM,WRV_AwM] = topn_recom(GroupRatingKMAwM_AA,topN,selectedItem);
                        [hit_AV,topn_out_AV,topn_idx_AV,WRV_AV] = topn_recom(GroupRatingKMAV_AA,topN,selectedItem);
                        [hit_MP,topn_out_MP,topn_idx_MP,WRV_MP] = topn_recom(GroupRatingKMMP_AA,topN,selectedItem);
                        [hit_LM,topn_out_LM,topn_idx_LM,WRV_LM] = topn_recom(GroupRatingKMLM_AA,topN,selectedItem);
                        [hit_MRP,topn_out_MRP,topn_idx_MRP,WRV_MRP] = topn_recom(GroupRatingKMMRP_AA,topN,selectedItem);

                        [ novelty_AVG,diversity_AVG ] = topn_Serendipity( topn_idx_Avg,similarityAll_item,Train_All,topN ,topn_idx_Avg);
                        [ novelty_AU,diversity_AU ] = topn_Serendipity( topn_idx_AU,similarityAll_item,Train_All,topN ,topn_idx_AU);
                        [ novelty_MUL,diversity_MUL ] = topn_Serendipity( topn_idx_MUL,similarityAll_item,Train_All,topN ,topn_idx_MUL);
                        [ novelty_SC,diversity_SC] = topn_Serendipity( topn_idx_SC,similarityAll_item,Train_All,topN ,topn_idx_SC);
                        [ novelty_AWM,diversity_AWM ] = topn_Serendipity( topn_idx_AwM,similarityAll_item,Train_All,topN ,topn_idx_AwM);
                        [ novelty_AV,diversity_AV] = topn_Serendipity( topn_idx_AV,similarityAll_item,Train_All,topN ,topn_idx_AV);
                        [ novelty_MP,diversity_MP ] = topn_Serendipity( topn_idx_MP,similarityAll_item,Train_All,topN ,topn_idx_MP);
                        [ novelty_LM,diversity_LM ] = topn_Serendipity( topn_idx_LM,similarityAll_item,Train_All,topN ,topn_idx_LM);
                        [ novelty_MRP,diversity_MRP ] = topn_Serendipity( topn_idx_MRP,similarityAll_item,Train_All,topN ,topn_idx_MRP);



                        hit_Cluster_AA(:,sayac2)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
                        novelty_Cluster_AA(:,sayac2)=[novelty_AVG;novelty_AU;novelty_MUL;novelty_SC;novelty_AWM;novelty_AV;novelty_MP;novelty_LM;novelty_MRP];
                        diversity_Cluster_AA(:,sayac2)=[diversity_AVG;diversity_AU;diversity_MUL;diversity_SC;diversity_AWM;diversity_AV;diversity_MP;diversity_LM;diversity_MRP];
                        WRV_Cluster_AA(:,sayac2)=[WRV_Avg;WRV_AU;WRV_MUL;WRV_SC;WRV_AwM;WRV_AV;WRV_MP;WRV_LM;WRV_MRP];
                        sayac2=sayac2+1;
                        %             else
                        %                 continue;
                        %             end


                    end
                    %% PUA Atak Modeline Göre Tasarım
%                     sayac3=1;
%                     for i=1:cluster_size
%                         %             if find(PA_uniqueCluster==i)
%                         Group_PA{i}=Pred_Train_PA(idxPA==i,:);
%                         GrupPA=Group_PA{1,i};
%                         GroupRatingKMMP_PA(sayac,1:size(GrupPA,2))=max(GrupPA);
%                         GroupRatingKMLM_PA(sayac,1:size(GrupPA,2))=min(GrupPA);
%                         [GroupRatingMRP] = mrp_Aggregation(GrupPA);
%                         GroupRatingKMMRP_PA(sayac,:)=GroupRatingMRP;
%                         for j=1:size(GrupPA,2)
%                             %                 if size(GrupPA,1)<10
%                             %                     GroupRatingKMAvg_PA(i,j)=0;
%                             %                     GroupRatingKMAU_PA(i,j)=0;
%                             %                     GroupRatingKMMUL_PA(i,j)=0;
%                             %                     GroupRatingKMSC_PA(i,j)=0;
%                             %                     GroupRatingKMAwM_PA(i,j)=0;
%                             %                     GroupRatingKMAV_PA(i,j)=0;
%                             %                 else
%                             GroupRatingKMAvg_PA(sayac,j)=sum(GrupPA(:,j))/nnz(GrupPA(:,j));
%                             GroupRatingKMAU_PA(sayac,j)=sum(GrupPA(:,j));
%                             GroupRatingKMMUL_PA(sayac,j)=prod(GrupPA(:,j));
%                             GroupRatingKMSC_PA(sayac,j)=nnz(GrupPA(:,j));
%                             GroupRatingKMAV_PA(sayac,j)=size(GrupPA(find(GrupPA(:,j)>=esikDeger),j),1);
%                             GroupRatingKMAwM_PA(sayac,j)=sum(GrupPA(find(GrupPA(:,j)>=esikDeger),j))/size(find(GrupPA(:,j)>=esikDeger),1);
%                             %                 end
%                         end
% 
%                         [hit_Avg,topn_out_Avg,topn_idx_Avg,WRV_Avg] = topn_recom(GroupRatingKMAvg_PA,topN,selectedItem);
%                         [hit_AU,topn_out_AU,topn_idx_AU,WRV_AU] = topn_recom(GroupRatingKMAU_PA,topN,selectedItem);
%                         [hit_MUL,topn_out_MUL,topn_idx_MUL,WRV_MUL] = topn_recom(GroupRatingKMMUL_PA,topN,selectedItem);
%                         [hit_SC,topn_out_SC,topn_idx_SC,WRV_SC] = topn_recom(GroupRatingKMSC_PA,topN,selectedItem);
%                         [hit_AwM,topn_out_AwM,topn_idx_AwM,WRV_AwM] = topn_recom(GroupRatingKMAwM_PA,topN,selectedItem);
%                         [hit_AV,topn_out_AV,topn_idx_AV,WRV_AV] = topn_recom(GroupRatingKMAV_PA,topN,selectedItem);
%                         [hit_MP,topn_out_MP,topn_idx_MP,WRV_MP] = topn_recom(GroupRatingKMMP_PA,topN,selectedItem);
%                         [hit_LM,topn_out_LM,topn_idx_LM,WRV_LM] = topn_recom(GroupRatingKMLM_PA,topN,selectedItem);
%                         [hit_MRP,topn_out_MRP,topn_idx_MRP,WRV_MRP] = topn_recom(GroupRatingKMMRP_PA,topN,selectedItem);
% 
%                         [ novelty_AVG,diversity_AVG ] = topn_Serendipity( topn_idx_Avg,similarityAll_item,Train_All,topN ,topn_idx_Avg);
%                         [ novelty_AU,diversity_AU ] = topn_Serendipity( topn_idx_AU,similarityAll_item,Train_All,topN ,topn_idx_AU);
%                         [ novelty_MUL,diversity_MUL ] = topn_Serendipity( topn_idx_MUL,similarityAll_item,Train_All,topN ,topn_idx_MUL);
%                         [ novelty_SC,diversity_SC] = topn_Serendipity( topn_idx_SC,similarityAll_item,Train_All,topN ,topn_idx_SC);
%                         [ novelty_AWM,diversity_AWM ] = topn_Serendipity( topn_idx_AwM,similarityAll_item,Train_All,topN ,topn_idx_AwM);
%                         [ novelty_AV,diversity_AV] = topn_Serendipity( topn_idx_AV,similarityAll_item,Train_All,topN ,topn_idx_AV);
%                         [ novelty_MP,diversity_MP ] = topn_Serendipity( topn_idx_MP,similarityAll_item,Train_All,topN ,topn_idx_MP);
%                         [ novelty_LM,diversity_LM ] = topn_Serendipity( topn_idx_LM,similarityAll_item,Train_All,topN ,topn_idx_LM);
%                         [ novelty_MRP,diversity_MRP ] = topn_Serendipity( topn_idx_MRP,similarityAll_item,Train_All,topN ,topn_idx_MRP);
% 
% 
% 
%                         hit_Cluster_PA(:,sayac3)=[hit_Avg;hit_AU;hit_MUL;hit_SC;hit_AwM;hit_AV;hit_MP;hit_LM;hit_MRP];
%                         novelty_Cluster_PA(:,sayac3)=[novelty_AVG;novelty_AU;novelty_MUL;novelty_SC;novelty_AWM;novelty_AV;novelty_MP;novelty_LM;novelty_MRP];
%                         diversity_Cluster_PA(:,sayac3)=[diversity_AVG;diversity_AU;diversity_MUL;diversity_SC;diversity_AWM;diversity_AV;diversity_MP;diversity_LM;diversity_MRP];
%                         WRV_Cluster_PA(:,sayac3)=[WRV_Avg;WRV_AU;WRV_MUL;WRV_SC;WRV_AwM;WRV_AV;WRV_MP;WRV_LM;WRV_MRP];
%                         sayac3=sayac3+1;
                        %             else continue;
                        %             end
%                     end

                    % C= TOPN DEĞERİDİR.
                    % RA
                    overall_hit_RA{1,c}=hit_Cluster_RA;
                    mean_hit_RA{1,c}=mean(mean(hit_Cluster_RA'));
                    agg_mean_hit_RA{1,c}=mean(hit_Cluster_RA');

                    overall_WRV_RA{1,c}=WRV_Cluster_RA;
                    mean_WRV_RA{1,c}=mean(mean(WRV_Cluster_RA'));
                    agg_mean_WRV_RA{1,c}=mean(WRV_Cluster_RA');

                    overall_novelty_RA{1,c}=novelty_Cluster_RA;
                    mean_novelty_RA{1,c}=mean(mean(novelty_Cluster_RA'));
                    agg_mean_novelty_RA{1,c}=mean(novelty_Cluster_RA');

                    overall_diversity_RA{1,c}=diversity_Cluster_RA;
                    mean_diversity_RA{1,c}=mean(mean(diversity_Cluster_RA'));
                    agg_mean_diversity_RA{1,c}=mean(diversity_Cluster_RA');

                    % AA
                    overall_hit_AA{1,c}=hit_Cluster_AA;
                    mean_hit_AA{1,c}=mean(mean(hit_Cluster_AA'));
                    agg_mean_hit_AA{1,c}=mean(hit_Cluster_AA');

                    overall_WRV_AA{1,c}=WRV_Cluster_AA;
                    mean_WRV_AA{1,c}=mean(mean(WRV_Cluster_AA'));
                    agg_mean_WRV_AA{1,c}=mean(WRV_Cluster_AA');

                    overall_novelty_AA{1,c}=novelty_Cluster_AA;
                    mean_novelty_AA{1,c}=mean(mean(novelty_Cluster_AA'));
                    agg_mean_novelty_AA{1,c}=mean(novelty_Cluster_AA');

                    overall_diversity_AA{1,c}=diversity_Cluster_AA;
                    mean_diversity_AA{1,c}=mean(mean(diversity_Cluster_AA'));
                    agg_mean_diversity_AA{1,c}=mean(diversity_Cluster_AA');

                    % PA
%                     overall_hit_PA{1,c}=hit_Cluster_PA;
%                     mean_hit_PA{1,c}=mean(mean(hit_Cluster_PA'));
%                     agg_mean_hit_PA{1,c}=mean(hit_Cluster_PA');
% 
%                     overall_WRV_PA{1,c}=WRV_Cluster_PA;
%                     mean_WRV_PA{1,c}=mean(mean(WRV_Cluster_PA'));
%                     agg_mean_WRV_PA{1,c}=mean(WRV_Cluster_PA');
% 
%                     overall_novelty_PA{1,c}=novelty_Cluster_PA;
%                     mean_novelty_PA{1,c}=mean(mean(novelty_Cluster_PA'));
%                     agg_mean_novelty_PA{1,c}=mean(novelty_Cluster_PA');
% 
%                     overall_diversity_PA{1,c}=diversity_Cluster_PA;
%                     mean_diversity_PA{1,c}=mean(mean(diversity_Cluster_PA'));
%                     agg_mean_diversity_PA{1,c}=mean(diversity_Cluster_PA');

                    %         overall_hit_PA_TR{t,c}=hit_Cluster_PA_TR;
                    %         mean_hit_PA_TR{t,c}=mean(hit_Cluster_PA_TR);

                end
                RA_kriteria_overallHIT{t,kriteria}=mean_hit_RA;
                RA_kriteria_aggHIT{t,kriteria}=agg_mean_hit_RA;
                RA_kriteria_overallWRV{t,kriteria}=mean_WRV_RA;
                RA_kriteria_aggWRV{t,kriteria}=agg_mean_WRV_RA;
                RA_kriteria_overallNOVELTY{t,kriteria}=mean_novelty_RA;
                RA_kriteria_aggNOVELTY{t,kriteria}=agg_mean_novelty_RA;
                RA_kriteria_overallDIVERSITY{t,kriteria}=mean_diversity_RA;
                RA_kriteria_aggDIVERSITY{t,kriteria}=agg_mean_diversity_RA;

                AA_kriteria_overallHIT{t,kriteria}=mean_hit_AA;
                AA_kriteria_aggHIT{t,kriteria}=agg_mean_hit_AA;
                AA_kriteria_overallWRV{t,kriteria}=mean_WRV_AA;
                AA_kriteria_aggWRV{t,kriteria}=agg_mean_WRV_AA;
                AA_kriteria_overallNOVELTY{t,kriteria}=mean_novelty_AA;
                AA_kriteria_aggNOVELTY{t,kriteria}=agg_mean_novelty_AA;
                AA_kriteria_overallDIVERSITY{t,kriteria}=mean_diversity_AA;
                AA_kriteria_aggDIVERSITY{t,kriteria}=agg_mean_diversity_AA;

%                 PA_kriteria_overallHIT{t,kriteria}=mean_hit_PA;
%                 PA_kriteria_aggHIT{t,kriteria}=agg_mean_hit_PA;
%                 PA_kriteria_overallWRV{t,kriteria}=mean_WRV_PA;
%                 PA_kriteria_aggWRV{t,kriteria}=agg_mean_WRV_PA;
%                 PA_kriteria_overallNOVELTY{t,kriteria}=mean_novelty_PA;
%                 PA_kriteria_aggNOVELTY{t,kriteria}=agg_mean_novelty_PA;
%                 PA_kriteria_overallDIVERSITY{t,kriteria}=mean_diversity_PA;
%                 PA_kriteria_aggDIVERSITY{t,kriteria}=agg_mean_diversity_PA;
            end
        end
        AA_overallHIT1=0;
        AA_overallHIT2=0;
        AA_overallHIT3=0;
        AA_overallHIT4=0;

        AA_overallWRV1=0;
        AA_overallWRV2=0;
        AA_overallWRV3=0;
        AA_overallWRV4=0;

        for i=1:1
            AA_overallHIT1=AA_overallHIT1+cell2mat(AA_kriteria_overallHIT{i,1});
            AA_overallHIT2=AA_overallHIT2+cell2mat(AA_kriteria_overallHIT{i,2});
            AA_overallHIT3=AA_overallHIT3+cell2mat(AA_kriteria_overallHIT{i,3});
            AA_overallHIT4=AA_overallHIT4+cell2mat(AA_kriteria_overallHIT{i,4});
        end

        for i=1:1
            AA_overallWRV1=AA_overallWRV1+cell2mat(AA_kriteria_overallWRV{i,1});
            AA_overallWRV2=AA_overallWRV2+cell2mat(AA_kriteria_overallWRV{i,2});
            AA_overallWRV3=AA_overallWRV3+cell2mat(AA_kriteria_overallWRV{i,3});
            AA_overallWRV4=AA_overallWRV4+cell2mat(AA_kriteria_overallWRV{i,4});
        end

        AA_overallAGG15=0;
        AA_overallAGG25=0;
        AA_overallAGG35=0;
        AA_overallAGG45=0;

        AA_overallAGG1_10=0;
        AA_overallAGG2_10=0;
        AA_overallAGG3_10=0;
        AA_overallAGG4_10=0;

        AA_overallAGG1_15=0;
        AA_overallAGG2_15=0;
        AA_overallAGG3_15=0;
        AA_overallAGG4_15=0;

        for i=1:1
            AA_overallAGG15=AA_overallAGG15+AA_kriteria_aggHIT{i,1}{1,1};
            AA_overallAGG25=AA_overallAGG25+AA_kriteria_aggHIT{i,2}{1,1};
            AA_overallAGG35=AA_overallAGG35+AA_kriteria_aggHIT{i,3}{1,1};
            AA_overallAGG45=AA_overallAGG45+AA_kriteria_aggHIT{i,4}{1,1};

            AA_overallAGG1_10=AA_overallAGG1_10+AA_kriteria_aggHIT{i,1}{1,2};
            AA_overallAGG2_10=AA_overallAGG2_10+AA_kriteria_aggHIT{i,2}{1,2};
            AA_overallAGG3_10=AA_overallAGG3_10+AA_kriteria_aggHIT{i,3}{1,2};
            AA_overallAGG4_10=AA_overallAGG4_10+AA_kriteria_aggHIT{i,4}{1,2};

            AA_overallAGG1_15=AA_overallAGG1_15+AA_kriteria_aggHIT{i,1}{1,3};
            AA_overallAGG2_15=AA_overallAGG2_15+AA_kriteria_aggHIT{i,2}{1,3};
            AA_overallAGG3_15=AA_overallAGG3_15+AA_kriteria_aggHIT{i,3}{1,3};
            AA_overallAGG4_15=AA_overallAGG4_15+AA_kriteria_aggHIT{i,4}{1,3};
        end

        AA_HIT=(AA_overallHIT1+AA_overallHIT2+AA_overallHIT3+AA_overallHIT4)/4;
        AA_WRV=(AA_overallWRV1+AA_overallWRV2+AA_overallWRV3+AA_overallWRV4)/4;
        AA_AGG_5=((AA_overallAGG15+AA_overallAGG25+AA_overallAGG35+AA_overallAGG45)/4)';
        AA_AGG_10=((AA_overallAGG1_10+AA_overallAGG2_10+AA_overallAGG3_10+AA_overallAGG4_10)/4)';
        AA_AGG_15=((AA_overallAGG1_15+AA_overallAGG2_15+AA_overallAGG3_15+AA_overallAGG4_15)/4)';

        agg_AA_all=[AA_AGG_5,AA_AGG_10,AA_AGG_15];
        RA_overallHIT1=0;
        RA_overallHIT2=0;
        RA_overallHIT3=0;
        RA_overallHIT4=0;
        RA_overallWRV1=0;
        RA_overallWRV2=0;
        RA_overallWRV3=0;
        RA_overallWRV4=0;

        for i=1:1
            RA_overallHIT1=RA_overallHIT1+cell2mat(RA_kriteria_overallHIT{i,1});
            RA_overallHIT2=RA_overallHIT2+cell2mat(RA_kriteria_overallHIT{i,2});
            RA_overallHIT3=RA_overallHIT3+cell2mat(RA_kriteria_overallHIT{i,3});
            RA_overallHIT4=RA_overallHIT4+cell2mat(RA_kriteria_overallHIT{i,4});
        end

        for i=1:1
            RA_overallWRV1=RA_overallWRV1+cell2mat(RA_kriteria_overallWRV{i,1});
            RA_overallWRV2=RA_overallWRV2+cell2mat(RA_kriteria_overallWRV{i,2});
            RA_overallWRV3=RA_overallWRV3+cell2mat(RA_kriteria_overallWRV{i,3});
            RA_overallWRV4=RA_overallWRV4+cell2mat(RA_kriteria_overallWRV{i,4});
        end
        RA_overallAGG15=0;
        RA_overallAGG25=0;
        RA_overallAGG35=0;
        RA_overallAGG45=0;

        RA_overallAGG1_10=0;
        RA_overallAGG2_10=0;
        RA_overallAGG3_10=0;
        RA_overallAGG4_10=0;

        RA_overallAGG1_15=0;
        RA_overallAGG2_15=0;
        RA_overallAGG3_15=0;
        RA_overallAGG4_15=0;

        for i=1:1
            RA_overallAGG15=RA_overallAGG15+RA_kriteria_aggHIT{i,1}{1,1};
            RA_overallAGG25=RA_overallAGG25+RA_kriteria_aggHIT{i,2}{1,1};
            RA_overallAGG35=RA_overallAGG35+RA_kriteria_aggHIT{i,3}{1,1};
            RA_overallAGG45=RA_overallAGG45+RA_kriteria_aggHIT{i,4}{1,1};

            RA_overallAGG1_10=RA_overallAGG1_10+RA_kriteria_aggHIT{i,1}{1,2};
            RA_overallAGG2_10=RA_overallAGG2_10+RA_kriteria_aggHIT{i,2}{1,2};
            RA_overallAGG3_10=RA_overallAGG3_10+RA_kriteria_aggHIT{i,3}{1,2};
            RA_overallAGG4_10=RA_overallAGG4_10+RA_kriteria_aggHIT{i,4}{1,2};

            RA_overallAGG1_15=RA_overallAGG1_15+RA_kriteria_aggHIT{i,1}{1,3};
            RA_overallAGG2_15=RA_overallAGG2_15+RA_kriteria_aggHIT{i,2}{1,3};
            RA_overallAGG3_15=RA_overallAGG3_15+RA_kriteria_aggHIT{i,3}{1,3};
            RA_overallAGG4_15=RA_overallAGG4_15+RA_kriteria_aggHIT{i,4}{1,3};
        end

        RA_HIT=(RA_overallHIT1+RA_overallHIT2+RA_overallHIT3+RA_overallHIT4)/4;
        RA_WRV=(RA_overallWRV1+RA_overallWRV2+RA_overallWRV3+RA_overallWRV4)/4;
        RA_AGG_5=((RA_overallAGG15+RA_overallAGG25+RA_overallAGG35+RA_overallAGG45)/4)';
        RA_AGG_10=((RA_overallAGG1_10+RA_overallAGG2_10+RA_overallAGG3_10+RA_overallAGG4_10)/4)';
        RA_AGG_15=((RA_overallAGG1_15+RA_overallAGG2_15+RA_overallAGG3_15+RA_overallAGG4_15)/4)';
        agg_RA_all=[RA_AGG_5,RA_AGG_10,RA_AGG_15];

        aaaa_AA_HitRatio{1,cB}=AA_HIT;
        aaaa_RA_HitRatio{1,cB}=RA_HIT;
        aaaa_AA_WRV{1,cB}=AA_WRV;
        aaaa_RA_WRV{1,cB}=RA_WRV;
        aaaa_AA_agg{1,cB}=agg_AA_all;
        aaaa_RA_agg{1,cB}=agg_RA_all;
    end
    aaaaSon_AA_HitRatio{1,mN}=aaaa_AA_HitRatio;
    aaaaSon_RA_HitRatio{1,mN}=aaaa_RA_HitRatio;
    aaaaSon_AA_WRV{1,mN}=aaaa_AA_WRV;
    aaaaSon_RA_WRV{1,mN}=aaaa_RA_WRV;
    aaaaSon_AA_agg{1,mN}=aaaa_AA_agg;
    aaaaSon_RA_agg{1,mN}=aaaa_RA_agg;
end
zzz_AA_001_HitRatio=[aaaaSon_AA_HitRatio{1,1}{1,1};aaaaSon_AA_HitRatio{1,2}{1,1};aaaaSon_AA_HitRatio{1,3}{1,1};aaaaSon_AA_HitRatio{1,4}{1,1}];
zzz_AA_005_HitRatio=[aaaaSon_AA_HitRatio{1,1}{1,2};aaaaSon_AA_HitRatio{1,2}{1,2};aaaaSon_AA_HitRatio{1,3}{1,2};aaaaSon_AA_HitRatio{1,4}{1,2}];
zzz_AA_01_HitRatio=[aaaaSon_AA_HitRatio{1,1}{1,3};aaaaSon_AA_HitRatio{1,2}{1,3};aaaaSon_AA_HitRatio{1,3}{1,3};aaaaSon_AA_HitRatio{1,4}{1,3}];

W_LAST_aa_Hitr=[zzz_AA_001_HitRatio;zzz_AA_005_HitRatio;zzz_AA_01_HitRatio];

zzz_RA_001_HitRatio=[aaaaSon_RA_HitRatio{1,1}{1,1};aaaaSon_RA_HitRatio{1,2}{1,1};aaaaSon_RA_HitRatio{1,3}{1,1};aaaaSon_RA_HitRatio{1,4}{1,1}];
zzz_RA_005_HitRatio=[aaaaSon_RA_HitRatio{1,1}{1,2};aaaaSon_RA_HitRatio{1,2}{1,2};aaaaSon_RA_HitRatio{1,3}{1,2};aaaaSon_RA_HitRatio{1,4}{1,2}];
zzz_RA_01_HitRatio=[aaaaSon_RA_HitRatio{1,1}{1,3};aaaaSon_RA_HitRatio{1,2}{1,3};aaaaSon_RA_HitRatio{1,3}{1,3};aaaaSon_RA_HitRatio{1,4}{1,3}];

W_LAST_Ra_Hitr=[zzz_RA_001_HitRatio;zzz_RA_005_HitRatio;zzz_RA_01_HitRatio];

zzz_AA_001_wrv=[aaaaSon_AA_WRV{1,1}{1,1};aaaaSon_AA_WRV{1,2}{1,1};aaaaSon_AA_WRV{1,3}{1,1};aaaaSon_AA_WRV{1,4}{1,1}];
zzz_AA_005_wrv=[aaaaSon_AA_WRV{1,1}{1,2};aaaaSon_AA_WRV{1,2}{1,2};aaaaSon_AA_WRV{1,3}{1,2};aaaaSon_AA_WRV{1,4}{1,2}];
zzz_AA_01_wrv=[aaaaSon_AA_WRV{1,1}{1,3};aaaaSon_AA_WRV{1,2}{1,3};aaaaSon_AA_WRV{1,3}{1,3};aaaaSon_AA_WRV{1,4}{1,3}];

W_LAST_aa_WRV=[zzz_AA_001_wrv;zzz_AA_005_wrv;zzz_AA_01_wrv];

zzz_RA_001_wrv=[aaaaSon_RA_WRV{1,1}{1,1};aaaaSon_RA_WRV{1,2}{1,1};aaaaSon_RA_WRV{1,3}{1,1};aaaaSon_RA_WRV{1,4}{1,1}];
zzz_RA_005_wrv=[aaaaSon_RA_WRV{1,1}{1,2};aaaaSon_RA_WRV{1,2}{1,2};aaaaSon_RA_WRV{1,3}{1,2};aaaaSon_RA_WRV{1,4}{1,2}];
zzz_RA_01_wrv=[aaaaSon_RA_WRV{1,1}{1,3};aaaaSon_RA_WRV{1,2}{1,3};aaaaSon_RA_WRV{1,3}{1,3};aaaaSon_RA_WRV{1,4}{1,3}];

W_LAST_Ra_WRV=[zzz_RA_001_wrv;zzz_RA_005_wrv;zzz_RA_01_wrv];

zzz_AA_001_AGG=[aaaaSon_AA_agg{1,1}{1,1},aaaaSon_AA_agg{1,2}{1,1},aaaaSon_AA_agg{1,3}{1,1},aaaaSon_AA_agg{1,4}{1,1}];
zzz_AA_005_AGG=[aaaaSon_AA_agg{1,1}{1,2},aaaaSon_AA_agg{1,2}{1,2},aaaaSon_AA_agg{1,3}{1,2},aaaaSon_AA_agg{1,4}{1,2}];
zzz_AA_01_AGG=[aaaaSon_AA_agg{1,1}{1,3},aaaaSon_AA_agg{1,2}{1,3},aaaaSon_AA_agg{1,3}{1,3},aaaaSon_AA_agg{1,4}{1,3}];

W_LAST_Aa_AGG=[zzz_AA_001_AGG;zzz_AA_005_AGG;zzz_AA_01_AGG];

zzz_RA_001_AGG=[aaaaSon_RA_agg{1,1}{1,1},aaaaSon_RA_agg{1,2}{1,1},aaaaSon_RA_agg{1,3}{1,1},aaaaSon_RA_agg{1,4}{1,1}];
zzz_RA_005_AGG=[aaaaSon_RA_agg{1,1}{1,2},aaaaSon_RA_agg{1,2}{1,2},aaaaSon_RA_agg{1,3}{1,2},aaaaSon_RA_agg{1,4}{1,2}];
zzz_RA_01_AGG=[aaaaSon_RA_agg{1,1}{1,3},aaaaSon_RA_agg{1,2}{1,3},aaaaSon_RA_agg{1,3}{1,3},aaaaSon_RA_agg{1,4}{1,3}];


W_LAST_RA_AGG=[zzz_RA_001_AGG;zzz_RA_005_AGG;zzz_RA_01_AGG];