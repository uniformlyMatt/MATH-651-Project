function [im2] = GetObjects(image, maxarea, threshold)

im = rgb2gray(image);

% Logical array for thresholding.
A = im > threshold;
im(~A) = 0;
im(A) = 255;

CC = bwconncomp(im);            % Find connected components within image.
S = regionprops(CC, 'Area');    % Compute area of each component.
L = labelmatrix(CC);            % Remove objects with area < maxarea.
im2 = 255*ismember(L, find([S.Area] >= maxarea));
im2 = bwmorph(im2, 'bridge', 2); % Bridge any almost connected pixels.

imwrite(255*im2, 'validation_objects.png');