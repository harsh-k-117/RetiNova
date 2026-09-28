function quality = assessQuality(img)
    gray = double(rgb2gray(img));
    
    % Build mask: all retinal tissue (main field + notch, since it's confirmed real retina)
    mask = gray > 10;
    mask = imfill(mask, 'holes');   % keeps notch's vessel-darkened pixels from creating gaps
    
    % Separate eroded mask ONLY for sharpness — avoids the genuine hard edge
    % at the tissue/background boundary from inflating the sharpness score
    sharpMask = imerode(mask, strel('disk', 15));
    
    lap = fspecial('laplacian');
    filtered = imfilter(gray, lap, 'replicate');
    sharpness = var(filtered(sharpMask));
    
    % Brightness/contrast/exposure use the full mask (not eroded) —
    % these aren't as sensitive to the boundary edge as sharpness is
    brightness   = mean(gray(mask));
    contrast     = std(gray(mask));
    overexposed  = sum(gray(mask) > 250) / sum(mask(:));
    underexposed = sum(gray(mask) < 5)  / sum(mask(:));
    
    quality.sharpness = sharpness;
    quality.brightness = brightness;
    quality.contrast = contrast;
    quality.overexposedFrac = overexposed;
    quality.underexposedFrac = underexposed;
    quality.isUsable = (sharpness > 7) && ...
                        (brightness > 40 && brightness < 140) && ...
                        (contrast > 10) && ...
                        ((overexposed + underexposed) < 0.01);
end