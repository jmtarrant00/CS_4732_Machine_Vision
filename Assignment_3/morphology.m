% Name: Jonathan Tarrant
% Number: 000-831-487
% Assignment 3

% Load image into program
image = imread("morphology.png");


%%% EROSION %%%

% Define the Structuring Element
structuring_element = strel('square', 5);


% Perform erosion on the image
eroded_image = imerode(image,structuring_element);


%%% DILATION %%%

% No need to redefine a structuring element 

% Perform Dilation on the image
dilated_image = imdilate(image, structuring_element);

% Display the images
subplot(2, 2, 1);
imshow(image);
title("Original Image")

subplot(2, 2, 2);
imshow(eroded_image);
title("Eroded Image");

subplot(2, 2, 3);
imshow(image);
title("Original Image");

subplot(2, 2, 4);
imshow(dilated_image);
title("Dilated Image");