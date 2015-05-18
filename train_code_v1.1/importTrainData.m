function data = importTrainData(file)

    %150955399 lines
    
    text = fgetl(file);
    if length(text) < 1
        data = [];
        fprintf('Out of Train Data!');
    end

    strVec = strsplit(text,',');
    data = str2double(strVec);

end


