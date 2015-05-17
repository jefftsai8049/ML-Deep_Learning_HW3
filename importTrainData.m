function data = importTrainData(fileName,endRow)

    %150955399 lines
    
    file = fopen(fileName);
    fseek(file, 0,'eof');
    fileSize = ftell(file);
    data = cell(endRow,1);
    % file = fopen('train_data/training_index_word2vec200.csv');
    file = fopen(fileName);
    i=1;
    while i < endRow+1
    % endCount=3;
    % for i=1:endCount
        text = fgetl(file);
        strVec = strsplit(text,',');
        mat = str2double(strVec);
        data(i) = mat2cell(mat);
        progress = i/endRow*100;
        fprintf('Load Training Data Progress : %.2f% \n',progress)
        i = i + 1;

    end

end


