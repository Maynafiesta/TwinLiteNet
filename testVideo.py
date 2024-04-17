import torch
import numpy as np
import cv2
import time
import threading
import socket
import queue
import pickle
from model import TwinLite as net
from tqdm.autonotebook import tqdm


class udpVideoStream:
    def __init__( self ) -> None:
        self.__hostIp = "192.168.1.14"
        self.__port = "10005"
        self.__bufferSize = 65536
        self.__streamSocket = None
        self.__hostName = None
        self.__stopFlag = False
        self.__frameQueue = queue.Queue( maxsize=10 )
        self.__streamThread = threading.Thread( target=self.__udpStream )

    def __initSocket( self ):
        self.__streamSocket = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )
        self.__streamSocket.setsockopt( socket.SOL_SOCKET, socket.SO_RCVBUF, self.__bufferSize )
        self.__hostName = socket.gethostname()
        socketAddr = ( self.__hostIp, self.__port )
        self.__streamSocket.bind( socketAddr )

    def __stopStream( self ) -> None:
        self.__stopFlag = True
        
    def __udpStream( self ):
        while not self.__stopFlag:
            pass

    def sendFrame( self, frame ):
        self.__frameQueue.put( item=pickle.dumps( frame ), block=True, timeout=1 )
        
    
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
    
    cv2.imshow( "DrivableArea -- LaneLines", cv2.hconcat([daFrame, llFrame])  )
    cv2.waitKey( 1 )
    
    return img_rs

if __name__ == "__main__":
    videoStreamer = udpVideoStream
    videoStreamer.mro
    
    model = net.TwinLiteNet()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('pretrained/best.pth'))
    model.eval()

    UDP_STREAM_URL = "udpsrc port=5000 " + \
        "! application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 " + \
        "! rtph264depay " + \
        "! decodebin " + \
        "! videoconvert " + \
        "! appsink"

    cap = cv2.VideoCapture( UDP_STREAM_URL, cv2.CAP_GSTREAMER )

    if not cap.isOpened():
        print( "Error: Could not open UDP stream." )
        exit()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print( "Error: Failed to receive frame from UDP stream." )
            break
        
        startTime = time.time()
        frameDetected = Run( model,frame )
        fpsVal = 1.0 / ( time.time() - startTime )
        frameTimeTagged = cv2.putText( frameDetected, str( fpsVal ), 
                                    ( 50, 50 ), cv2.FONT_HERSHEY_SIMPLEX , 1, 
                                    ( 255, 0, 0 ), 1, cv2.LINE_AA)
        streamQueue.put( frameTimeTagged )
        # cv2.imshow( 'Frame', frameTimeTagged )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
