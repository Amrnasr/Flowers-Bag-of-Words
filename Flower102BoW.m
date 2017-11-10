


% alpine sea holly	43	buttercup	buttercup	71	fire lily	fire lily	40
% anthurium	anthurium	105	californian poppy	californian poppy	102	foxglove	foxglove	162
% artichoke	artichoke	78	camellia	camellia	91	frangipani	frangipani	166
% azalea	azalea	96	canna lily	canna lily	82	fritillary	fritillary	91
% ball moss	ball moss	46	canterbury bells	canterbury bells	40	garden phlox	garden phlox	45
% balloon flower	balloon flower	49	cape flower	cape flower	108	gaura	gaura	67
% barbeton daisy	barbeton daisy	127	carnation	carnation	52	gazania	gazania	78
% bearded iris	bearded iris	54	cautleya spicata	cautleya spicata	50	geranium	geranium	114
% bee balm	bee balm	66	clematis	clematis	112	giant white arum lily	giant white arum lily	56
% bird of paradise	bird of paradise	85	colt's foot	colt's foot	87	globe thistle	globe thistle	45
% bishop of llandaff	bishop of llandaff	109	columbine	columbine	86	globe-flower	globe-flower	41
% black-eyed susan	black-eyed susan	54	common dandelion	common dandelion	92	grape hyacinth	grape hyacinth	41
% blackberry lily	blackberry lily	48	corn poppy	corn poppy	41	great masterwort	great masterwort	56
% blanket flower	blanket flower	49	cyclamen 	cyclamen	154	hard-leaved pocket orchid	hard-leaved pocket orchid	60
% bolero deep blue	bolero deep blue	40	daffodil	daffodil	59	hibiscus	hibiscus	131
% bougainvillea	bougainvillea	128	desert-rose	desert-rose	63	hippeastrum 	hippeastrum	76
% bromelia	bromelia	63	english marigold	english marigold	65	japanese anemone	japanese anemone	55
% king protea	king protea	49	peruvian lily	peruvian lily	82	stemless gentian	stemless gentian	66
% lenten rose	lenten rose	67	petunia	petunia	258	sunflower	sunflower	61
% lotus	lotus	137	pincushion flower	pincushion flower	59	sweet pea	sweet pea	56
% love in the mist	love in the mist	46	pink primrose	pink primrose	40	sweet william	sweet william	85
% magnolia	magnolia	63	pink-yellow dahlia?	pink-yellow dahlia?	109	sword lily	sword lily	130
% mallow	mallow	66	poinsettia	poinsettia	93	thorn apple	thorn apple	120
% marigold	marigold	67	primula	primula	93	tiger lily	tiger lily	45
% mexican aster	mexican aster	40	prince of wales feathers	prince of wales feathers	40	toad lily	toad lily	41
% mexican petunia	mexican petunia	82	purple coneflower	purple coneflower	85	tree mallow	tree mallow	58
% monkshood	monkshood	46	red ginger	red ginger	42	tree poppy	tree poppy	62
% moon orchid	moon orchid	40	rose	rose	171	trumpet creeper	trumpet creeper	58
% morning glory	morning glory	107	ruby-lipped cattleya	ruby-lipped cattleya	75	wallflower	wallflower	196
% orange dahlia	orange dahlia	67	siam tulip	siam tulip	41	water lily	water lily	194
% osteospermum	osteospermum	61	silverbush	silverbush	52	watercress	watercress	184
% oxeye daisy	oxeye daisy	49	snapdragon	snapdragon	87	wild pansy	wild pansy	85
% passion flower	passion flower	251	spear thistle	spear thistle	48	windflower	windflower	54
% pelargonium	pelargonium	71	spring crocus	spring crocus	42	yellow iris	


% 
% % yellow iris	49
% alpine sea holly	43
% buttercup	71
% fire lily	40
% anthurium	105
% californian poppy	102
% foxglove	162
% artichoke	78
% camellia	91
% frangipani	166
% azalea	96
% canna lily	82
% fritillary	91
% ball moss	46
% canterbury bells	40
% 
% garden phlox	45
% 
% balloon flower	49
% 
% cape flower	108
% 
% gaura	67
% 
% 


clc; clear all; close all;



outputFolder2='E:\HthmWork\Computer-Vision\Projects\Matlab-MachineLearning_Amr_17-02-20\Matlab-MachineLearning\Image Classification\';

rootFolder = fullfile(outputFolder2, '102flowers');


% Get a list of all files and folders in this folder.
files = dir('E:\102flowers')
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir]
% Extract only those that are directories.
subFolders = files(dirFlags)
% Print folder names to command window.
for k = 1 : length(subFolders)
	fprintf('Sub folder #%d = %s\n', k, subFolders(k).name);
end

% Construct an ImageDatastore based on the following
% categories from Caltech 101: 'airplanes', 'ferry', 'laptop'. 
% Use imageDatastore to help you manage the data. Since imageDatastore 
% operates on image file locations, and therefore does not load all 
% the images into memory, it is safe to use on large image collections.

categories = {'1  pink primrose', '2 hard-leaved pocket orchid', '3 bolero deep blue'};
i=4;
categories{1,i}='4 giant white arum lily';

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

% % You can easily inspect the number of images per category
% % as well as category labels as shown below:

tbl = countEachLabel(imds)

% Prepare Training and Validation Image Sets
% Since imds above contains an unequal number of images per category, let's first adjust it, so that the number of images in the training set is balanced.

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
% ans =
% 
%   3×2 table
% 
%       Label      Count
%     _________    _____
% 
%     airplanes    67   
%     ferry        67   
%     laptop       67   
% 
% Separate the sets into training and validation data. Pick 30% of images from each set for the training data and the remainder, 70%, for the validation data. Randomize the split to avoid biasing the results.

[trainingSet, validationSet] = splitEachLabel(imds, 0.3, 'randomize');

% Find the first instance of an image for each category
pinkprimrose= find(trainingSet.Labels == '1  pink primrose', 1);
 pocketorchid= find(trainingSet.Labels == '2 hard-leaved pocket orchid', 1);
bolerodeepblue = find(trainingSet.Labels == '3 bolero deep blue', 1);

% figure

subplot(1,3,1);
imshow(readimage(trainingSet,pinkprimrose))
subplot(1,3,2);
imshow(readimage(trainingSet,pocketorchid))
subplot(1,3,3);
imshow(readimage(trainingSet,bolerodeepblue))


% Create a Visual Vocabulary and Train an Image Category Classifier
% Bag of words is a technique adapted to computer vision from the world of natural language processing. Since images do not actually contain discrete words, we first construct a "vocabulary" of SURF features representative of each image category.
% 
% This is accomplished with a single call to bagOfFeatures function, which:
% 
% extracts SURF features from all images in all image categories
% constructs the visual vocabulary by reducing the number of features through quantization of feature space using K-means clustering

bag = bagOfFeatures(trainingSet);


% Additionally, the bagOfFeatures object provides an encode method for counting the visual word occurrences in an image. It produced a histogram that becomes a new and reduced representation of an image.

img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')


% This histogram forms a basis for training a classifier and for the actual image classification. In essence, it encodes an image into a feature vector.
% 
% Encoded training images from each category are fed into a classifier training process invoked by the trainImageCategoryClassifier function. Note that this function relies on the multiclass linear SVM classifier from the Statistics and Machine Learning Toolbox™.

categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);



% Evaluate Classifier Performance
% Now that we have a trained classifier, categoryClassifier, let's evaluate it. As a sanity check, let's first test it with the training set, which should produce near perfect confusion matrix, i.e. ones on the diagonal.

confMatrix = evaluate(categoryClassifier, trainingSet);


% Next, let's evaluate the classifier on the validationSet, which was not used during the training. By default, the evaluate function returns the confusion matrix, which is a good initial indicator of how well the classifier is performing.

confMatrix = evaluate(categoryClassifier, validationSet);

% Compute average accuracy
mean(diag(confMatrix));


h = msgbox('Open a flower image for program to classify'); 

disp('Press any key');
pause(10);
[f,p] = uigetfile({'*.png;*.jpg;*.bmp;*.tif','Supported images';...
                 '*.png','Portable Network Graphics (*.png)';...
                 '*.jpg','J-PEG (*.jpg)';...
                 '*.bmp','Bitmap (*.bmp)';...
                 '*.tif','Tagged Image File (*.tif,)';...
                 '*.*','All files (*.*)'});
x = imread([p f]);
himage=imshow(x);
title(f)
img=x;

% Try the Newly Trained Classifier on Test Images
% You can now apply the newly trained classifier to categorize new images.

% img = imread(fullfile(rootFolder, '1  pink primrose', 'image_0690.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)













