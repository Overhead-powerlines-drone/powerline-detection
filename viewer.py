from logging import raiseExceptions
import numpy as np
import cv2
import argparse
import os.path
import time

#realsense library
import pyrealsense2 as rs

#custom library
from utils import imageprocessing as imgprocess
from utils import line

#argparse
parser = argparse.ArgumentParser(description='Power lines detection ')
parser.add_argument('--input', default='cam', type=str, help='input can be cam by using --input cam or --input bag for .bag file')
parser.add_argument('--filename', default=0, type=str, help='file path')
parser.add_argument('--record', default=0, type=int, help='0 = disable recording, 1 = enable recording')

args = parser.parse_args()

#Color class for terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

#setup based on init args
if args.input == 'cam':

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    if args.record == 1:

        if not args.filename:
            print('filename parameter is not given, use --filename (path)')
            exit()
        
        if os.path.splitext(args.filename)[1] != ".bag":
            print('file must be .bag format')
            exit()
        
        config.enable_record_to_file(args.filename)

    
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break

    print('Found Device: ' + bcolors.OKGREEN + f'{device_product_line}' + bcolors.ENDC)

    if found_rgb:
        print('RGB Color: ' + bcolors.OKGREEN + f'{found_rgb}' + bcolors.ENDC)
    else:
        print('RGB Color: ' + bcolors.WARNING + f'{found_rgb}' + bcolors.ENDC)
elif args.input == 'bag':

    if not args.filename:
        print('filename parameter is not given, use --filename (path)')
        exit()


    if os.path.splitext(args.filename)[1] != ".bag":
        print('file must be .bag format')
        exit()

    if not os.path.exists(args.filename):
        print('file does not exists')
        exit()

    rs.config.enable_device_from_file(config, args.filename)
    pass
else:
    print('Unknow input: ' + bcolors.WARNING + f'{args.input}' + bcolors.ENDC)
    exit()

#alignment for depth to RGB Image
align_to = rs.stream.color
align = rs.align(align_to)

def main():

    #setting up the pipeline based on the config
    profile = pipeline.start(config)

    #retriving the depth_scale from camera for unit convertion
    sensor = profile.get_device().first_depth_sensor()
    depth_scale = sensor.get_depth_scale()

    #fps changes very frequent per frame so taking the average is better
    fps_list = []
    fps = 0

    #save count keeps track of how many images we saved
    save_count = 0

    #main loop
    while True:

        #RETERIVE DATA ------------------------------------------------ 
        start = time.perf_counter()
        #wait for frame to be available
        frames = pipeline.wait_for_frames()

        #We wanna make sure the depth view is aligned with camera view
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        #depth & color images are stored in numpy array with the same resolution
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        #DATA PROCESSING ------------------------------------------------ 

        #clip pixels that are 10 meters away. For drone we might wanna reduce it to 5 meters 
        clipped_image = imgprocess.clip_based_distance(color_image, depth_image, depth_scale, 10)

        #find edges
        edge_image = imgprocess.EdgeDetecion(clipped_image)

        #get lines from edge image
        lines = line.HoughLines(edge_image)
        line_objects = line.filter_lines(edge_image, lines)
        #lines = utils.filter_lines(lines)
        #GUI PROCESSING -------------------------------------------------

        #depth map has one channel, therefore we will convert it to rgb and use color map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_INFERNO)

        #draw obtained lines
        # for line in lines:
        #     cv2.line(color_image, (line.x1, line.y1), (line.x2, line.y2), (255, 100, 100), 5, cv2.LINE_AA)

        #draw line objects
        cv2.drawContours(color_image, line_objects, -1, (0,255,0), 3)

        #draw center of a contours:
        for c in line_objects:

            cx, cy = line.get_contour_center(c)
            depth_pixel = imgprocess.get_depth_from_pixels(cx, cy, depth_image, depth_scale)
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(color_image, f"{depth_pixel}m", (cx,cy + 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)

        #put text for each image
        cv2.putText(color_image, "RGB", (20,70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_colormap, "DEPTH", (20,70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(clipped_image, "CLIPPED", (20,70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(edge_image, "EDGES", (20,70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2, cv2.LINE_AA)

        #stacking 4 of the images in one image
        top_images = np.hstack((color_image, depth_colormap))
        bot_images = np.hstack((clipped_image, np.dstack((edge_image,edge_image,edge_image))))


        stacked_images = np.vstack((top_images, bot_images))

        #put fps text in the center of the final image
        cv2.putText(stacked_images, f"FPS:{fps}", (int(stacked_images.shape[1] / 2) - 150, int(stacked_images.shape[0] / 2)), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2, cv2.LINE_AA)

        #DISPLAY ---------------------------------------------------------
        cv2.namedWindow('Viewer', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Viewer', stacked_images)

        k = cv2.waitKey(1)
        if k == ord('q'):
          break
        elif k == ord('a'):
            imgprocess.save_frame(stacked_images, f"save{save_count}")
            save_count += 1

        period = time.perf_counter() - start

        #cacluating fps
        fps_list.append(1 / period)

        fps = round(sum(fps_list) / len(fps_list))

        if len(fps_list) > 30:
            fps_list.pop(0)

    pipeline.stop()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == '__main__':
    main()