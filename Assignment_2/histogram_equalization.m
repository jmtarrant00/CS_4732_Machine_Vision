% Name: Jonathan Tarrant
% Number: 000831487
% Project 3

% Histogram Equalization

banker = imread("banker.jpeg");

input_histogram = imhist(banker);

equalized_image = histeq(banker);

equalized_image_histogram = imhist(equalized_image);

uneq_mean = mean2(banker);
uneq_std = std2(double(banker));
eq_mean = mean2(equalized_image);
eq_std = std2(double(equalized_image));

subplot(2, 2, 1);
imshow(banker);
title(['Unequalized Mean: ', num2str(eq_mean)]);

subplot(2, 2, 2);
bar(input_histogram);
title(['Unequalized Std Dev: ', num2str(uneq_std)])

subplot(2, 2, 3);
imshow(equalized_image);
title(['Equalized Mean: ', num2str(eq_mean)])

subplot(2, 2, 4);
bar(equalized_image_histogram);
title(['Equalized Std Dev: ', num2str(eq_std)])
