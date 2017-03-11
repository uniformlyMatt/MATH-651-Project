clear;

% Read in the binary image.
im = imread('training_image.png');
im = 255*im;

% Display the image.
colormap(gray(256)); image(im); axis image;

outpath = 'D:/MATH651/MATH-651-Project/TrainingSet/subimages/';

% Count and label every object in the image. This is only possible since
% the image is binary, with a mask applied to known buildings.
[labeledImage, num_obj] = bwlabel(im);
[m, n] = size(im);

% Find centroids of all objects in labeled image.
cent = regionprops(labeledImage,'centroid');

% Set the size of the sub-images.
sub_size = 128;

% This loop creates a sub-image of size sub_size x sub_size and saves it as
% a png.
for i = 1:num_obj
    % Extracts each building separately.
    thisBlob = ismember(labeledImage, i) > 0;
    indiv_building = im;
    indiv_building(~thisBlob) = 0;
    
    % Find the centroid of the ith object.
    y = floor(cent(i).Centroid(1));
    x = floor(cent(i).Centroid(2));
    
    if (y == 0)
        y = 1;
    end;
    if (x == 0)
        x = 1;
    end;
    
    % Set the bounds for extracting the image so that the sub-image is
    % relatively centered.
    if (x - sub_size/2 < 1)
        x_ind = 1;
    elseif (x + sub_size/2 > m)
        x_ind = m - (sub_size + 1);
    else
        x_ind = x - sub_size/2;
    end;
    
    if (y - sub_size/2 < 1)
        y_ind = 1;
    elseif (y + sub_size/2 > n)
        y_ind = n - (sub_size + 1);
    else
        y_ind = y - sub_size/2;    
    end;
    
%     disp([num2str(x_ind), ' is the upper left corner.'])
%     disp([num2str(y_ind), ' is the upper right corner.'])
    subImage = indiv_building(x_ind:x_ind + (sub_size-1), y_ind:y_ind + (sub_size-1));
    imwrite(subImage, [outpath,'subImage',num2str(i),'.png']) 
end;