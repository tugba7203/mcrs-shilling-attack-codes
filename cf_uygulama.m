function [predict_VeriSeti] = cf_uygulama(Veriseti,enyakinkomsu)
simVeriSeti=similarity_measurePCC(Veriseti,'user');
for i=1:size(Veriseti,1)
    [deger indis]=sort(simVeriSeti(i,:),'descend');
    indis(find(isnan(deger)==1))=[];
    deger(find(isnan(deger)==1))=[];
    deger(find(indis==i))=[];
    indis(find(indis==i))=[];
    
    nnKomsuDeger=deger(1:enyakinkomsu);
    nnKomsuIndis=indis(1:enyakinkomsu);
    for j=1:size(Veriseti,2)
%         if Veriseti(i,j)==0
            payToplam=0;
            for t=1:enyakinkomsu
                payToplam=payToplam+((Veriseti(nnKomsuIndis(t),j)-(sum(Veriseti(nnKomsuIndis(t),:))/nnz(Veriseti(nnKomsuIndis(t),:))))*simVeriSeti(i,nnKomsuIndis(t)));
                %payToplam=payToplam+((Veriseti(nnKomsuIndis(t),j)-(mean(Veriseti(nnKomsuIndis(t),:))))*simVeriSeti(i,nnKomsuIndis(t)));
            end
            predict_VeriSeti(i,j)=(sum(Veriseti(i,:))/nnz(Veriseti(i,:)))+(payToplam/sum(simVeriSeti(i,nnKomsuIndis)));
            %predict_VeriSeti(i,j)=(mean(Veriseti(i,:)))+(payToplam/sum(simVeriSeti(i,nnKomsuIndis)));

%         else
%             predict_VeriSeti(i,j)=Veriseti(i,j);

%         end
    end
end
end