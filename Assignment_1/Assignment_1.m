% Load the image
image = imread('rose.jpg');

% Downsample the image to different sizes
downsampled_512 = customResize(image, 512, 512);
downsampled_256 = customResize(image, 256, 256);
downsampled_128 = customResize(image, 128, 128);

% Upsample the images back to 1024x1024
upsampled_512 = customResize(downsampled_512, 1024, 1024);
upsampled_256 = customResize(downsampled_256, 1024, 1024);
upsampled_128 = customResize(downsampled_128, 1024, 1024);

% Create a figure to display images
figure;

% Display the original image
subplot(3, 3, 1);
imshow(image);
title('Original Image');

% Display down-sampled images
subplot(3, 3, 2);
imshow(downsampled_512);
title('Downsampled 512x512');
subplot(3, 3, 3);
imshow(downsampled_256);
title('Downsampled 256x256');
subplot(3, 3, 4);
imshow(downsampled_128);
title('Downsampled 128x128');

% Display up-sampled images
subplot(3, 3, 5);
imshow(upsampled_128);
title('Upsampled 128x128');
subplot(3, 3, 6);
imshow(upsampled_256);
title('Upsampled 256x256');
subplot(3, 3, 7);
imshow(upsampled_512);
title('Upsampled 512x512');


% Adjust figure layout
sgtitle('Image Down-Sampling and Up-Sampling');

% Custom resizing function
function output = customResize(input, new_height, new_width)
    [orig_height, orig_width, ~] = size(input);
    row_scale = orig_height / new_height;
    col_scale = orig_width / new_width;
    
    output = zeros(new_height, new_width, size(input, 3), 'uint8');
    
    for r = 1:new_height
        for c = 1:new_width
            orig_r = round(r * row_scale);
            orig_c = round(c * col_scale);
            
            % Ensure indices are within bounds
            orig_r = min(max(orig_r, 1), orig_height);
            orig_c = min(max(orig_c, 1), orig_width);
            
            output(r, c, :) = input(orig_r, orig_c, :);
        end
    end
end