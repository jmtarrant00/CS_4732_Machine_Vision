% Load the image
image = imread('rose.jpg');

% Convert to double for calculations
image_double = double(image);

% Change the gray levels
gray_128 = uint8(round(image_double * (128/256)));
gray_64 = uint8(round(image_double * (64/256)));
gray_32 = uint8(round(image_double * (32/256)));

% Display the images
figure;
subplot(2, 2, 1);
imshow(image);
title('Original Image');
subplot(2, 2, 2);
imshow(gray_128);
title('Gray Levels: 128');
subplot(2, 2, 3);
imshow(gray_64);
title('Gray Levels: 64');
subplot(2, 2, 4);
imshow(gray_32);
title('Gray Levels: 32');
sgtitle('Gray Levels');
