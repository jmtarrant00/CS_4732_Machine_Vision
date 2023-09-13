% Name: Jonathan Tarrant
% Number: 000831487
% Project 2
close all;
clear;
clc;

% Read in the image
fourier = imread("fourierspectrum.pgm");

% Convert it to a double
fourier = double(fourier);


%%% LOG TRANSFORM %%%

% Apply the log transform to the image
log_transformed_image = log_transform(100, fourier);

% Convert images back to uint8 for display
fourier = uint8(fourier);
log_transformed_image = uint8(log_transformed_image);

% Display Images
subplot(2,2,1);
imshow(fourier);
title("Pre Log Transform")
subplot(2,2,2);
imshow(log_transformed_image);
title("Post Log Transform")


%%% POWER LAW TRANSFORM %%%

% Convert Fourier back to double
fourier = double(fourier);

% Apply the power law transformation
power_law_transformed_image = power_law_transform(1, 1.5, fourier);

% Convert Images to uint8 for display
fourier = uint8(fourier);
power_law_transformed_image = uint8(power_law_transformed_image);

% Display Images
subplot(2,2,3);
imshow(fourier);
title("Pre Power Law");
subplot(2,2,4);
imshow(power_law_transformed_image);
title("Post Power Law");

% Function for log transform
function s = log_transform(c, r)
    s = c*log(1+r);
end

function s = power_law_transform(c, gamma, r)
    s = c*(r.^gamma);
end