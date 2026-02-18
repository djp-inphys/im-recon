% Initialize a cell array to store the text elements
lbl = cell(225,1);

idx = 1;
for phase = 1:7
    for row = 1:15
        for col = 1:15
            % Create a text element for the current row and column
            string = num2str(row) +  "," + num2str(col)
            lbl{idx} = char( string );
            idx = idx + 1;
        end
    end
end