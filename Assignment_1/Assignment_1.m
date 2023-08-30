% Assignment 1 
% Jonathan Tarrant
% CS 4732

% import the image 
I = imread("Assignment_1\rose.jpg");
% Set list of sizes to resize to 
sizes = [512, 256, 128];


figure;

% Get the current size of the image
[m, n] = size(I);
subplot(1, length(sizes) + 1, 1);
imshow(uint8(I));
title([num2str(m) ' x ' num2str(n)]);

I = double(I);

[X, Y] = meshgrid(1:m, 1:n);
for newSizeIndex = 1:length(sizes)
    newSize = sizes(newSizeIndex);
    xi = linspace(1, m, newSize);
    yi = linspace(1, n, newSize);
    [Xi, Yi] = meshgrid(xi, yi);

    Ii = interp2(X, Y, I, Xi, Yi);

    subplot(1, length(sizes) + 1, newSizeIndex + 1)
    imshow(uint8(Ii));
    title(['Resized to ' num2str(newSize) ' x ' num2str(newSize)]);
end

