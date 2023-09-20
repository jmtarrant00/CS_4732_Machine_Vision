% Name: Jonathan Tarrant
% Number: 000-831-487
% Assignment 3

% Read the image into the program
cells_image = imread("cell.jpg");

% Greyscale the image
grey_cells = rgb2gray(cells_image);

% Apply Thresholding
threshold_image = imbinarize(grey_cells);

% Apply Dilation to the image to make sure all connected cell parts
structural = strel('disk', 10);
closed_image = imclose(threshold_image, structural);

% Find the connected componenets (count the cells)
connected_components = bwconncomp(closed_image);
disp(connected_components)

% Find the biggest cell
cell_sizes = cellfun(@numel, connected_components.PixelIdxList);
[~, biggest_cell_index] = max(cell_sizes);

% Get the boundary Image of the largest cell
boundary_image = false(size(closed_image));
boundary_image(connected_components.PixelIdxList{biggest_cell_index}) = true;


% Show all the steps above
subplot(3, 2, 1);
imshow(cells_image);
title('Original Image')

subplot(3, 2, 2);
imshow(grey_cells);
title("Greyed Out Image")

subplot(3, 2, 3);
imshow(threshold_image);
title("Threshold function ")

subplot(3, 2, 4);
imshow(closed_image);
title("Morphological Closing")

subplot(3, 2, 5);
imshow(boundary_image);
title("Largest Cell")

% Print the number of cells 
disp(['Number of Cells: ', num2str(connected_components.NumObjects)])
