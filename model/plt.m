close all;
colors = ["r.", "g.", "b.", "c.", "m.", "y.", "k."];

r_start = 1;
r_end   = 225;
for phase = 1:2
    scatter3( kl_scores(r_start:r_end,1), kl_scores(r_start:r_end,2), kl_scores(r_start:r_end,3), colors(phase));

    hold("on"); grid("on");
    r_start = r_start + 225;
    r_end   = r_end + 225;
end
    
% Add arrows connecting paired points
r_start = 1;
r_end   = 225;
points1 = [kl_scores(r_start:r_end,1), kl_scores(r_start:r_end,2), kl_scores(r_start:r_end,3)];

r_start = r_start + 225;
r_end   = r_end + 225;
points2 = [kl_scores(r_start:r_end,1), kl_scores(r_start:r_end,2), kl_scores(r_start:r_end,3)];


for i = 1:size( points1, 1)
    line([points1(i, 1), points2(i, 1)], ...
         [points1(i, 2), points2(i, 2)], ...
         [points1(i, 3), points2(i, 3)], 'Color', 'b');
%     quiver3(points1(i, 1), points1(i, 2), points1(i, 3), ...
%                 points2(i, 1) - points1(i, 1), points2(i, 2) - points1(i, 2), points2(i, 3) - points1(i, 3), 'b');
end