% Name: Jonathan Tarrant
% Number: 000-831-487
% Assignment 3

% Read in the image into the program
fingerprint = imread("fingerprint_BW.png");


%%% MORPHOLOGICAL FILTERING %%%

% Define the Structuring Element
structuring_element = strel('square', 3);

% Perform Erosion on the image
morph_filtered_image = imerode(fingerprint, structuring_element);

% Perform Dilation on image
morph_filtered_image = imdilate(morph_filtered_image, structuring_element);


%%% MEDIAN FILTERING %%%

% Convert fingerprint image to double
fingerprint_double = double(fingerprint);

% Define the neighborhood for median filtering
neighborhood = [3, 3, 3];

% Apply Median Filtering
median_filtered_image = medfilt3(fingerprint_double, neighborhood);


% Show everything in a window
subplot(2, 2, 1);
imshow(fingerprint);
title("Original Image")

subplot(2, 2, 3);
imshow(morph_filtered_image);
title("Morphological Filtering");

subplot(2, 2, 4);
imshow(median_filtered_image);
title("Median Filtering");

