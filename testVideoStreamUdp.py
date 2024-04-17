import cv2
import gi

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

# Define the video file path
video_file = "your_video.mp4"  # Replace with your video file path

# Define the target IP address and port
host = "127.0.0.1"  # Replace with the receiver's IP address
port = 5000

def gstreamer_pipeline(width, height):
  return f"appsrc name=source ! videoconvert ! video/x-raw,format=(string)BGR,width={width},height={height} ! x264enc tune=zerolatency speed-preset=ultrafast ! rtph264pay config-interval=10 pt=96 ! udpsink host={host} port={port}"

def on_error(bus, msg):
  err = msg.parse_error()
  print(f"Error: {err.message}")
  loop.quit()

def main():
  # Initialize GStreamer
  Gst.init(None)

  # Open video capture with OpenCV
  cap = cv2.VideoCapture(video_file)

  # Check if video capture opened successfully
  if not cap.isOpened():
    print("Error opening video file!")
    return

  # Get video frame width and height
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Create the GStreamer pipeline based on video resolution
  pipeline = gstreamer_pipeline(width, height)

  # Create the GStreamer pipeline object
  stream = Gst.parse_launch(pipeline)

  # Add a bus watch to catch errors
  bus = stream.get_bus()
  bus.add_signal_watch()
  bus.connect("message", on_error)

  # Main loop to capture, encode, and send video frames
  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
      print("Error reading frame!")
      break

    # Convert frame to a Gst.Buffer (Optional, might be handled by videoconvert)
    # buffer = Gst.Buffer.new_allocate(None, len(frame.flatten()), None)
    # buffer.fill(frame.flatten().astype(np.uint8))

    # (Optional) Push the frame to the appsrc element (if used)
    # appsrc = stream.get_by_name("source")
    # appsrc.push_buffer(buffer)

    # Start streaming (can be moved outside the loop for continuous streaming)
    stream.set_state(Gst.State.PLAYING)

    # Process the frame here (if needed)

    # Release the GStreamer buffer (if used)
    # buffer.unref()

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    # Stop streaming
    stream.set_state(Gst.State.NULL)

  # Clean up resources
  cap.release()
  Gst.Object.unref(stream)

if __name__ == "__main__":
  main()