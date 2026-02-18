function [feat] = reframe3( data )

nrm_flag = true;
n_phases = 7;
n_channels = 64;

[r, c] = size( data );

% Initialize a cell array to store the text elements
fprintf( "here!\n")
try
    % Check if the original matrix has a number of rows that is a multiple of 7
    if mod(size(data, 1), n_phases) ~= 0
        error('The number of rows in the original matrix is not a multiple of 7.');
    end
    if size(data, 2) ~= n_channels
        error('The number of cols in the original matrix is not equal to the number of channels i.e. 64.');
    end
    
    % Initialize the reshaped matrix
    feat = zeros(r/7, n_channels*n_phases);
    
    % Reshape the matrix
    for idx = 1:size( feat, 1)
        rows = data((idx-1)*7+1:idx*7, :);
%         feat(i, :) = rows(:)';
        feat(idx, :) = reshape(rows.', 1, []);
    end
catch me
    fprintf(me.identifier)
end

