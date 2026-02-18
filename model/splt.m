% im(:,:,1) = model2(1:120,1:120);
% im(:,:,2) = model2(120+1:2*120,1:120);
% im(:,:,3) = model2(120*2+1:3*120,1:120);
% im(:,:,4) = model2(120*3+1:4*120,1:120);
% im(:,:,5) = model2(120*4+1:5*120,1:120);
% im(:,:,6) = model2(120*5+1:6*120,1:120);
% im(:,:,7) = model2(120*6+1:7*120,1:120);
% 
im = [];
for phase = 1:7
    idx = 1;
    for row = 1:15
        for col = 1:15
            r_start = (row-1)*64+1;
            r_end = r_start + 63;
            im(idx, :, phase) = model3(r_start:r_end, col)';
            idx = idx + 1;
        end
    end
end

im = [];
idx = 1;
for phase = 1:7
    for row = 1:15
        for col = 1:15
            r_start = (row-1)*64+1;
            r_end = r_start + 63;
            im(idx, :) = model3(r_start:r_end, col)';
            idx = idx + 1;
        end
    end
end
