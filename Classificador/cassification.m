% Universidade Federal de Goias
% Aluno: Carlos Henrique Rorato Souza - 201600718
%
% Classificacao de imagens astronomicas utilizando
% uma rede neural convolucional profunda (ResNet50)
 
% Criando o Image Data Store  e calculando valores iniciais
rootFolder = fullfile('data');
categories = {'Galaxia','Nebulosa','Planeta'};

imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds,minSetCount,'randomize');
countEachLabel(imds);

% Pre-processamento das imagens (filtros)
numberOfImages = length(imds.Files);
for k = 1 : numberOfImages
    inputFileName = imds.Files{k};
    fprintf('Pre-processando imagem %s\n', inputFileName);

    img_ori = imread(inputFileName);
    
    %Filtros de media e mediana para remocao de ruido
    filtro = fspecial('average', [2 2]);
    img_mediana = medfilt3(imfilter(img_ori, filtro));
    img_mediana = medfilt3(img_mediana, [5 5 5]);
    
    %salvando a imagem final
    imwrite(img_mediana,inputFileName);
end

% Mostrando algumas imagens contidas no IMDS
Galaxia = find(imds.Labels == 'Galaxia',1);
Nebulosa = find(imds.Labels == 'Nebulosa',1);
Planeta = find(imds.Labels == 'Planeta',1);

figure;
subplot(2,2,1);
imshow(readimage(imds, Galaxia));
subplot(2,2,2);
imshow(readimage(imds, Nebulosa));
subplot(2,2,3);
imshow(readimage(imds, Planeta));

% Carrega a rede ResNet50
net = resnet50();

% Mostrando a arquitetura da rede
figure;
plot(net);
title('Arquitetura da Rede ResNet-50');
set(gca,'YLim',[150 170]);

net.Layers(1);
net.Layers(end);

numel(net.Layers(end).ClassNames);

% Fazendo o split no imds, considerando percentual de 60 porcento das imagens
% para treinamento
[trainingSet, testSet] = splitEachLabel(imds, 0.6, 'randomize');

imageSize = net.Layers(1).InputSize;

% Organizando o tamanho das imagens a serem utilizadas
augmentedTrainingSet = augmentedImageDatastore(imageSize, ...
    trainingSet, 'ColorPreprocessing', 'gray2rgb');

augmentedTestSet = augmentedImageDatastore(imageSize, ...
    testSet, 'ColorPreprocessing', 'gray2rgb');

% Mostrando os pessos iniciais da rede, em forma de imagem
w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);

figure;
montage(w1);
title('Pesos da Primeira Camada Convolucional');

% Trabalhando com a camada de caracteristicas e criando o classificador
featureLayer = 'fc1000';
trainingFeatures = activations(net, ...
    augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32,'OutputAs','columns');

trainingLabels = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, ...
    augmentedTestSet, featureLayer, 'MiniBatchSize', 32,'OutputAs','columns');

% Fazendo os testes na rede e calculando a acuracia da rede
predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns' );

testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat,2));

disp('Acuracia da rede: ');
disp(mean(diag(confMat)));

%Salvar as variaveis
save('classifier.mat', 'classifier');
save('imageSize.mat', 'imageSize');
save('featureLayer.mat', 'featureLayer');

disp('Variaveis salvas com sucesso!');


