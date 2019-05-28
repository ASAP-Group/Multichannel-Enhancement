function [ ] = saveMatFile4Troch( fName , data )

    fid = fopen( fName ,'wb');
    fwrite(fid,size(data,1)*size(data,2),'uint32');
    fwrite(fid,data,'float');
    fclose(fid);     

end

