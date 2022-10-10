from dataclasses import dataclass
import numpy as np
import cv2
import math

class imageprocessing:

    def get_depth_from_pixels(x, y, depth, depth_scale):
        """
        Note:This gets neighbor pixel's depth and average them. unlike the function below which only get depth of one pixel.
        Args:
            x: x value of the pixel
            y: y value of the pixel
            depth: the depth map
            depthscale: depth scale given by the camera
        Returns:
            depth measurment in meters
        """
        pixels_depth = []
        scan_size = 5
        h = depth.shape[0]
        w = depth.shape[1]

        startPixel = [x - scan_size, y - scan_size]
        endPixel = [x + scan_size, y + scan_size]

        if(startPixel[0] < 0):
            startPixel[0] = 0
        
        if(startPixel[1] < 0):
            startPixel[1] = 0

        if(endPixel[0] > w):
            endPixel[0] = w
        
        if(endPixel[1] > h):
            endPixel[1] = h

        for x in range(startPixel[0], endPixel[0]):

            for y in range(startPixel[1], endPixel[1]):

                pixels_depth.append(depth[y, x])

        average_depth = 0
        for x in pixels_depth:
            average_depth += x

        average_depth = average_depth / len(pixels_depth)

        average_depth = average_depth * depth_scale

        return round(average_depth, 2)

    def get_distance_from_pixel(x, y, depth, depth_scale):
        """
        Args:
            x: x value of the pixel
            y: y value of the pixel
            depth: the depth map
            depthscale: depth scale given by the camera
        Returns:
            depth measurment in meters
        """
        pixel_depth = depth[y][x]
        pixel_depth = pixel_depth * depth_scale

        return round(pixel_depth, 2)

    def clip_based_distance(image, depth, depth_scale, clip_distance = 10, clipped_color = 0):
        """
        args:
            image: Colored image from intel realsense
            depth: Depth image from the intel realsense
            depth_scale: the scale for converting depth data to meters
            clip_distance: distance in meter
            clipped_color: clipped pixel color
        return:
            clipped image (numpy)
        """
        assert image.shape != depth.shape, 'Error at clip_based_distance: image and depth shapes dont match'

        clip_distance = clip_distance / depth_scale

        depth_image = np.dstack((depth,depth,depth))
        clipped_image = np.where((depth_image > clip_distance) | (depth_image <= 0), clipped_color, image)

        return clipped_image

    def EdgeDetecion(image):
        """
        args:
            image: grayscale
        return:
            binary image of edges
        """
        image = cv2.GaussianBlur(image,(5,5),0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.Canny(image, 50, 200, None, 7)

        return image
    
    def save_frame(image, filename = 'save'):
        """
        args:
            image: image to save
            filename: file name, no need for .png
        """
        try:
            cv2.imwrite('save'+'.png', image)
        except:
            print('Failed to save image')

@dataclass
class line:
    """
    Store x1, x2, y1, y2 for line segmentation.
    provide line segmentation method for calculations
    """
    x1: int
    y1: int
    x2: int
    y2: int

    def getLength(self, line):
        """
        return: the length between itself and another line.
        """
        point1 = self.center()
        point2 = line.center()

        return round(math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2), 2)


    def angle(self):
        """
        return: the angle of a line with respect to the x-axis
        """
        m = (self.y2 - self.y1) / (self.x2 - self.x1)

        return math.atan(m)

    def center(self):
        """
        return: the center point of a line
        """
        return (self.x2 - self.x1) / 2, (self.y2 - self.y1) / 2

    def HoughLines(image, threshold = 350):
        """
        args:
            image: a binary image with edge detection

        return:
            list of lines, containing the start point and end point of a line
        """
        lines = []
        hlines = cv2.HoughLines(image, 1, np.pi / 180, threshold, None, 0, 0)
        if hlines is not None:
            for i in range(0, len(hlines)):
                rho = hlines[i][0][0]
                theta = hlines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho

                lines.append(line(int(x0 + 1000*(-b)), int(y0 + 1000*(a)),int(x0 - 1000*(-b)),int(y0 - 1000*(a))))

        return lines

    def get_contour_center(contour):
        """
        Return: x, y center location of a contourt
        """

        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY

    def filter_lines(image, lines):
        """
        args:
            image: for resolution only
            lines: line cluster generated by hough transform
        return:
            a list of contours
        """
        blank_image = np.zeros((image.shape[0],image.shape[1],1), dtype=np.uint8)

        for line in lines:
            cv2.line(blank_image, (line.x1, line.y1), (line.x2, line.y2), (255, 255, 255), 5, cv2.LINE_AA)

        contours, hierarchy = cv2.findContours(blank_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours