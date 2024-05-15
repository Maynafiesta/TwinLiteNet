import torch
import numpy as np
import cv2
import os
import sys
from model import TwinLite as net
import shutil

def Run( model,img ):
    img = cv2.resize( img, ( 640, 360 ) )
    img_rs = img.copy()

    img = img[:, :, ::-1].transpose( 2, 0, 1 )
    img = np.ascontiguousarray( img )
    img = torch.from_numpy( img )
    img = torch.unsqueeze( img, 0 )  # add a batch dimension
    img = img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model( img )
    x0=img_out[0]
    x1=img_out[1]

    _,da_predict = torch.max( x0, 1 )
    _,ll_predict = torch.max( x1, 1 )

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    img_rs[DA > 100] = [255 ,0, 0]
    img_rs[LL > 100] = [0, 255, 0]
    
    daFrame = 255 * np.zeros( shape=img_rs.shape, dtype=np.uint8 )
    daFrame[DA > 100] = [255, 255, 255]
    
    llFrame = 255 * np.zeros( shape=img_rs.shape, dtype=np.uint8 )
    llFrame[LL > 100] = [255, 255, 255]
    
    # cv2.imshow( "DrivableArea -- LaneLines", cv2.hconcat([daFrame, llFrame])  )
    # cv2.waitKey( 1 )
    
    # return img_rs
    return img_rs, daFrame, llFrame
    

def printUsage():
    # TODO
    return 0

if __name__ == "__main__":
    sourceDir = "-"
    # destDir = "-"

    argList = sys.argv
    argListSize = len( argList )
    
    if argListSize > 0:
        for i in range( 1, len( argList ) ):
            if( ( "-h" == argList[i] ) or ( "--help" == argList[i] ) ):
                printUsage( argList[0] )
                sys.exit( 0 )

            elif( ( "-sd" == argList[i] ) or ( "--sourcedir" == argList[i] ) ):
                if( ( argListSize >= i + 1 ) and ( "-" == argList[i + 1][0] ) ):
                    print( "Invalid path! Path could not start with '-'." )
                    sys.exit(1)

                sourceDir = argList[i + 1]
                i = i + 1

    # if( ( "-" == destDir ) or ( "-" == sourceDir ) ):
    if "-" == sourceDir:
        print( "Source and dist should be given." )
        sys.exit

    model = net.TwinLiteNet()
    model = torch.nn.DataParallel( model )
    model = model.cuda()
    model.load_state_dict( torch.load( 'pretrained/best.pth' ) )
    model.eval()

    image_list = os.listdir( sourceDir )
    destDir = sourceDir + "/results/"
    destDirAll = destDir + "all/"
    destDirDA = destDir + "DA/"
    destDirLL = destDir + "LL/"
    
    if ( not os.path.exists( destDirAll) ):
        os.mkdir( destDir )
        os.mkdir( destDirAll )
        os.mkdir( destDirDA )
        os.mkdir( destDirLL )
        

    for i, imgName in enumerate( image_list ):
        if os.path.isdir( sourceDir + imgName  ):
            print( "Not File => ", sourceDir + "/" + imgName )
            continue

        print( "image:", ( destDir + imgName ) )
        img = cv2.imread( os.path.join( sourceDir, imgName ) )
        img, daFrame, llFrame = Run( model, img )
        # cv2.imwrite( os.path.join( destDir, imgName ), img )
        cv2.imwrite( ( destDirAll +  "DL_" +imgName ), img )
        cv2.imwrite( ( destDirDA +  "DA_" +imgName ), daFrame )
        cv2.imwrite( ( destDirLL +  "LL_" +imgName ), llFrame )
        