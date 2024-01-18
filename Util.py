from queue import Queue
from vmbpy import *


class Handler:

    def __init__(self):
        self.display_queue = Queue(10)

    def get_image(self):
        return self.display_queue.get(True)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)

            self.display_queue.put(frame.as_opencv_image(), True)

        cam.queue_frame(frame)
