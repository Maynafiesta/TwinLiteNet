import torch
import numpy as np
import torch
from model import TwinLite as net
import cv2
import time

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
    
    return img_rs

if __name__ == "__main__":
        
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel( model )
    model = model.cuda()
    model.load_state_dict(torch.load( 'pretrained/best.pth' ) )
    model.eval()

    _WIDTH = 1920
    _HEIGHT = 512
    _FPS = 25
    # _IP_ADDR = "127.0.0.1"
    _IP_ADDR = "224.1.1.101"
    _PORT = 5002

    UDP_LISTENER_URL = "udpsrc port=5000 " + \
        "! application/x-rtp, " + \
            "media=video, clock-rate=90000, encoding-name=H264, payload=96 " + \
        "! rtph264depay " + \
        "! decodebin " + \
        "! videoconvert " + \
        "! videoscale " + \
        "! videocrop " + \
            "top=300 " + \
            "left=0 " + \
            "right=0 " + \
            "bottom=300 " + \
        "! appsink"

    UDP_STREAMER_URL = "appsrc " + \
        "! video/x-raw " + \
            ", format=BGR " + \
            ", framerate=" + str( _FPS ) + "/1 " + \
            ", width=" + str( _WIDTH ) + " " + \
            ", height=" + str( _HEIGHT ) + " " + \
        "! queue " + \
        "! videoconvert " + \
        "! x264enc tune=zerolatency " + \
        "! rtph264pay config-interval=10 pt=96 " + \
        "! udpsink host=" + _IP_ADDR + " port=" + str( _PORT ) + " "

    cap = cv2.VideoCapture( UDP_LISTENER_URL, cv2.CAP_GSTREAMER )
    writer = cv2.VideoWriter( UDP_STREAMER_URL, cv2.CAP_GSTREAMER, 0, _FPS, ( _WIDTH, _HEIGHT ) )

    if not cap.isOpened():
        print( "Error: Could not open UDP listener." )
        exit()

    if not writer.isOpened():
        print( "Error: Could not open UDP streamer." )
        exit()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print( "Error: Failed to receive frame from UDP stream." )
            break
        
        startTime = time.time()
        # Left frame[0:480, 0:640] - Middle = frame[0:480, 640:1280] - Right =  frame[0:480, 1280:1920]
        frameDetected = cv2.hconcat([ Run( model, frame[0:480, 0:640] ), 
                                    Run( model, frame[0:480, 640:1280] ),
                                    Run( model, frame[0:480, 1280:1920] )] )
        frameDetected = cv2.resize( frameDetected, ( _WIDTH, _HEIGHT ) )
        fpsVal = 1.0 / ( time.time() - startTime )
        frameTimeTagged = cv2.putText( frameDetected, str( fpsVal ),
                                    ( 50, 50 ), cv2.FONT_HERSHEY_SIMPLEX , 1,
                                    ( 255, 0, 0 ), 1, cv2.LINE_AA )
        writer.write( frameTimeTagged )

        # cv2.imshow( 'FramePython', frameTimeTagged )
        # if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
        #     break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

