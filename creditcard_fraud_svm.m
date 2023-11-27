% Veriyi oku csv dosyası
veri = readtable('creditcard.csv');

% Veriyi train ve test setlerine ayır
rng(1); % Çoğul tekrarlanabilirlik için rastgele sayı üreteciyi ayarla
idx = randperm(size(veri, 1));
train_size = round(0.8 * size(veri, 1));
train_veri = veri(idx(1:train_size), :);
test_veri = veri(idx(train_size+1:end), :);

% K-fold sayısı
kfold_num = 5;

% K-fold çapraz doğrulama ile modeli eğit ve değerlendir
cv = cvpartition(size(train_veri, 1), 'KFold', kfold_num);
f1_scores = zeros(kfold_num, 1);
accuracies = zeros(kfold_num, 1);

for fold = 1:kfold_num
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);

    subtrain_veri = train_veri(train_idx, :);
    subtest_veri = train_veri(test_idx, :);

    % Destek vektör makineleri (SVM) modelini oluştur ve eğit
    svm_model = fitcsvm(subtrain_veri{:, 2:end-1}, subtrain_veri{:, end}, 'Standardize', true);

    % Subtest setinde modelin performansını değerlendir
    subtest_etiketler = predict(svm_model, subtest_veri{:, 2:end-1});
    
    % Confusion matrix oluştur
    cm = confusionmat(subtest_veri{:, end}, subtest_etiketler);

    % F1 score ve accuracy hesapla
    precision = cm(2,2) / sum(cm(:,2));
    recall = cm(2,2) / sum(cm(2,:));
    f1_score = 2 * (precision * recall) / (precision + recall);
    accuracy = sum(diag(cm)) / sum(cm(:));

    f1_scores(fold) = f1_score;
    accuracies(fold) = accuracy;

    fprintf('Fold %d - F1 Score: %.4f, Accuracy: %.4f\n', fold, f1_score, accuracy);
end

% Kullanılan kfold sayısı boyunca F1 Score ve Accuracy'nin ortalamasını hesapla
ortalama_f1_score = mean(f1_scores);
ortalama_accuracy = mean(accuracies);
fprintf('Ortalama F1 Score: %.4f, Ortalama Accuracy: %.4f\n', ortalama_f1_score, ortalama_accuracy);

% Eğitilen modeli test setine uygula ve başarıyı hesapla
test_etiketler = predict(svm_model, test_veri{:, 2:end-1});

% Confusion matrix oluştur
cm_test = confusionmat(test_veri{:, end}, test_etiketler);

% F1 score ve accuracy hesapla
precision_test = cm_test(2,2) / sum(cm_test(:,2));
recall_test = cm_test(2,2) / sum(cm_test(2,:));
f1_score_test = 2 * (precision_test * recall_test) / (precision_test + recall_test);
accuracy_test = sum(diag(cm_test)) / sum(cm_test(:));

fprintf('Test seti için F1 Score: %.4f, Accuracy: %.4f\n', f1_score_test, accuracy_test);
