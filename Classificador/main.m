% Universidade Federal de Goias
% Aluno: Carlos Henrique Rorato Souza - 201600718
%
% Classificacao de imagens astronomicas utilizando
% uma rede neural convolucional profunda (ResNet50)

% carrega a rede ResNet50
net = resnet50();

% carregar as variaveis
classifier = load('var/classifier.mat');
imageSize = load('var/imageSize.mat');
featureLayer = load('var/featureLayer.mat');

classifier = classifier.classifier;
imageSize = imageSize.imageSize;
featureLayer = featureLayer.featureLayer;

% ---- CARREGUE A IMAGEM AQUI
newImage = imread('testes/teste5.jpg');

% pre-processamento da imagem
newImage2 = newImage;
filtro = fspecial('average', [2 2]);
newImage = medfilt3(imfilter(newImage, filtro));
newImage = medfilt3(newImage, [5 5 5]);

% segmentacao da imagem por limiarizacao
bw = im2bw(newImage, 0.15); 
st = regionprops(bw, 'BoundingBox', 'Area' );
for ii= 1 : length(st)
    Areai(ii)= st(ii).Area;
end
largest_blob_id = find(Areai==max(Areai));

% encontra todas as areas maiores ou iguais a decima parte da maior area
limiar = ceil(st(largest_blob_id).Area/10);
x = 1;
for ii = 1 : length(st)
    if(st(ii).Area >= limiar) 
        indexSubImage(x)= ii;
        x = x + 1;
    end
end

% mostra alguns resultados
figure;
imshow(newImage);
title('Imagem Pre-Processada'); 

figure;
imshow(bw);
title('Imagem Binarizada');    

% mostrando a imagem original com a classificacao das areas em destaque
figure;
imshow(newImage2);
hold;

% Para cada area encontrada, cria uma subimagem e classifica-a
for kk = 1 : length(indexSubImage)
    subImage = imcrop(newImage, [st(indexSubImage(kk)).BoundingBox(1),st(indexSubImage(kk)).BoundingBox(2),st(indexSubImage(kk)).BoundingBox(3),st(indexSubImage(kk)).BoundingBox(4)]);
    
    % classificando a imagem
    ds = augmentedImageDatastore(imageSize, ...
        subImage, 'ColorPreprocessing', 'gray2rgb');

    imageFeatures = activations(net, ...
        ds, featureLayer, 'MiniBatchSize', 32,'OutputAs','columns');

    label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns' );
    
    % criando o retangulo verde na imagem final
    rectangle('Position',[st(indexSubImage(kk)).BoundingBox(1),st(indexSubImage(kk)).BoundingBox(2),st(indexSubImage(kk)).BoundingBox(3),st(indexSubImage(kk)).BoundingBox(4)], 'EdgeColor','g','LineWidth',1.5 );
    text(st(indexSubImage(kk)).BoundingBox(1)+5,st(indexSubImage(kk)).BoundingBox(2)+10,char(label), 'Color','g','FontSize',12);
    hold;
end

title('Imagem Classificada');