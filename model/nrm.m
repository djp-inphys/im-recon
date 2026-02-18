function [n_lines] = nrm( lines )

[model_size, n_channels, n_phases] = size( lines );
fprintf( "here!\n")
try
    sum_3 = sum( lines, 3);

    n_lines = lines./sum_3;
catch me
    fprintf(me.identifier)
end