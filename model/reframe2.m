function [lines] = reframe2( model )

nrm_flag = true;

modelT = model'; 
% fud_modelT = flipud(modelT);
model = modelT;
N_CHANNELS = 64;
N_PHASES = 7;
% Initialize a cell array to store the text elements
fprintf( "here!\n")
m_size = length(model)/64;
lines = [];
sub_model = zeros(N_PHASES, N_CHANNELS);
try
    for modidx = 1:m_size
        for channel = 1:N_CHANNELS
            index = modidx + (channel-1)*m_size;
            sub_model(:, channel) = model(:, index);
        end

        if (nrm_flag == true)
            sub_model = sub_model./sum(sub_model);
        end

        lines = [lines;sub_model];
    end
catch me
    fprintf(me.identifier)
end

