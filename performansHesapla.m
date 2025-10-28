function [grupNDCG,grupFairness,grupSatisfaction] = performansHesapla(GroupRating,Group,idx,TrainO,k,esikDeger,selectedItem)
mOran=3.5;
for n=1:4
    N=5*n;
    for nd=1:k
        nDCG=[];
        fairness=0;
        satisfaction=0;

        [vGD iGD]=sort(GroupRating(nd,:),'descend');
        topNindis=iGD(1:N);
        topNdeger=vGD(1:N);
        Groups=cell2mat(Group(1,nd));
        if size(Groups,1)<10
            continue;
        else
            grupKullaniciIndis=find(idx==nd);
            for gi=1:size(grupKullaniciIndis,1)
                toplamDCG=0;
                toplamIDCG=0;
                gercekDegerU=TrainO(grupKullaniciIndis(gi,1),:);
                [vUGD iUGD]=sort(gercekDegerU(topNindis),'descend');
                for ndc=2:N
                    toplamDCG=toplamDCG+(gercekDegerU(1,topNindis(1,ndc))/log2(ndc));
                    toplamIDCG=toplamIDCG+(vUGD(1,ndc)/log2(ndc));
                end
                dcg=gercekDegerU(1,topNindis(1,1))+toplamDCG;
                idcg=vUGD(1,1)+toplamIDCG;
                nDCG(gi,1)=dcg/idcg;
                if isnan(nDCG(gi,1))
                    nDCG(gi,1)=0;
                end
                if isinf(nDCG(gi,1))
                    nDCG(gi,1)=0;
                end
                if(size(find(vUGD>=mOran),2)>0)
                    fairness=fairness+1;
                end
                %             satisfaction=satisfaction+(size(find(vUGD>=esikDeger),2));
                satisfaction=satisfaction+(size(find(Groups(gi,topNindis)>=esikDeger),2));
            end
        end
        grupNDCG(nd,n)=mean(nDCG);
        grupFairness(nd,n)=fairness/size(grupKullaniciIndis,1);
        grupSatisfaction(nd,n)=satisfaction/(size(grupKullaniciIndis,1)*N);

    end
end
if ((size(grupNDCG,1)<k)==1)
    grupNDCG((size(grupNDCG,1)+1:k),:)=0;
    grupFairness((size(grupFairness,1)+1:k),:)=0;
    grupSatisfaction((size(grupSatisfaction,1)+1:k),:)=0;
end
end