function [optimal_k] = elbowMethod(data)
% Denenecek küme sayıları
k_values = 1:10;

% SSE'yi saklamak için dizi
sse_values = zeros(size(k_values));

% K-means algoritmasını farklı küme sayıları için çalıştır
for i = 1:length(k_values)
    k = k_values(i);
    [~, C, sumd] = kmeans(data, k);
    sse_values(i) = sum(sumd);
end

% SSE'yi gösteren grafik
figure;
plot(k_values, sse_values, 'o-');
title('Elbow Method - Optimal K Determination');
xlabel('Number of Clusters (k)');
ylabel('Sum of Squared Errors (SSE)');

% Dirsek noktasını belirlemek için kullanıcıdan yardım alınabilir
prompt = 'Enter the elbow point from the graph: ';
elbow_point = input(prompt);

% Kullanıcının seçtiği dirsek noktasına göre optimal kümeyi seç
optimal_k = k_values(elbow_point);

disp(['Optimal Cluster Number (k): ', num2str(optimal_k)]);
end