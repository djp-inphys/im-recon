function [lines] = reframe( model )

N_CHANNELS = 64;
N_PHASES = 7;
% Initialize a cell array to store the text elements
fprintf( "here!\n")
m_size = length(model)/64;
lines = zeros(m_size, N_CHANNELS, N_PHASES);
try
    for channel = 1:N_CHANNELS
        for modidx = 1:m_size
            for phase = 1:N_PHASES
                % Create a text element for the current row and column
                index = modidx + (channel-1)*m_size;
                if (index <= length( model ))
                    lines(modidx, channel, phase) = model(index, phase);
                else
                    fprintf( "out of bounds\n");
                end
            end
        end
    end
catch me
    fprintf(ME.identifier)
end