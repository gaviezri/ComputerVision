import cv2 as cv
import numpy as np
from os import getcwd

# Define constants for lane switching direction
RIGHT = 2
LEFT = 1

def slope(x, y, w, z, frame):
    """
    Calculates the slope of a line defined by two points.
    :param x: x-coordinate of the first point
    :param y: y-coordinate of the first point
    :param w: x-coordinate of the second point
    :param z: y-coordinate of the second point
    :param frame: the frame to which the line belongs
    :return: the slope of the line
    """
    return (y - z) / (x - w)


class LaneDetector():
    def __init__(self):
        self.frame_processed = 0
        # Store coordinates of the previous frame's lines
        # 0 = r_low , 1 = r_up , 2 = l_low , 3 = l_up
        self.prev_lines_coords = [(900, 625), (655, 415), (223, 415), (537, 625)]
        # range of right_lower_x values representing static driving
        self.right_lane_anchor_TH = (950, 1170)
        # frame counter for printing message
        self.message_frame_counter = 0
        # 0 = false / 1 = to the LEFT / 2 = to the RIGHT
        self.lane_switching = 0

    def drawLane(self, lines, frame):
        """
        Draws the lane on the input frame
        :param lines: lines to be used to draw the lane
        :param frame: the frame on which the lane is to be drawn
        :return: the input frame with the lane drawn on it
        """
        # getting lane coordinates
        # every ~0.5 a sec choose new lines (videowriter set to 25fps)
        if self.frame_processed % 11 == 1:
            self.prev_lines_coords = self.chooseTwoLines(lines)

        right_lower, right_upper, left_lower, left_upper = self.prev_lines_coords
        mask = np.zeros_like(frame)
        # lane mask
        cv.fillPoly(mask, np.array([[left_lower, left_upper, right_upper, right_lower]], dtype=np.int32),
                    (255, 255, 255))
        # making the road between lines darker
        overlay = frame.copy()
        overlay[mask != 0] = 40
        # making the highlighted lanes with opacity
        alpha = 0.4
        frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv.line(frame, right_lower, right_upper, (0, 0, 255), thickness=3)
        cv.line(frame, left_lower, left_upper, (0, 0, 255), thickness=3)

        return frame

    def getMaskVertices(self, processed_frame):
        """
        Generates the vertices of the mask according to the current lane switching status
        :param processed_frame: the current frame that is being processed
        :return: the vertices of the mask as an array
        """
        # generating vertices according to current lane switching status
        if self.lane_switching == 0:
            return np.array([[
                (125, processed_frame.shape[0] - 100),
                (550, 430),
                (750, 430),
                (processed_frame.shape[1] - 125, processed_frame.shape[0] - 100)]],
                dtype=np.int32)
        elif self.lane_switching == LEFT:
            return np.array([[
                (50, processed_frame.shape[0] - 100),
                (50, 430),
                (750, 430),
                (processed_frame.shape[1] - 125, processed_frame.shape[0] - 100)]],
                dtype=np.int32)
        else:
            return np.array([[
                (125, processed_frame.shape[0] - 100),
                (550, 430),
                (processed_frame.shape[1] - 50, 430),
                (processed_frame.shape[1] - 50, processed_frame.shape[0] - 100)]],
                dtype=np.int32)

    def createMask(self, processed_frame):
        """
        Creates a mask for the current frame according to the current lane switching status
        :param processed_frame: the current frame that is being processed
        :return: the masked frame
        """
        vertices = self.getMaskVertices(processed_frame)

        mask = np.zeros_like(processed_frame)
        # plot the trapeze on the mask
        cv.fillPoly(mask, vertices, (255, 255, 255))

        return cv.bitwise_and(processed_frame, mask)

    def laneSwitchDetection(self, frame):
        """
        Detects if the car is switching lanes and displays a message on the frame
        :param frame: the current frame
        :return: the input frame with the message added
        """
        cur_right_low = self.prev_lines_coords[0][0]

        if cur_right_low in range(self.right_lane_anchor_TH[0], self.right_lane_anchor_TH[1]):
            self.lane_switching = 0
            return frame

        self.message_frame_counter += 1
        message = "Changing lanes to the "

        if self.message_frame_counter <= 80:

            # changing lane to the RIGHT
            if cur_right_low < self.right_lane_anchor_TH[0]:
                message += "RIGHT!"
                self.lane_switching = RIGHT

            elif cur_right_low > self.right_lane_anchor_TH[1]:
                message += "LEFT!"
                self.lane_switching = LEFT

            # adding the message to the frame
            frame = cv.putText(frame, message, (70, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        else:
            self.message_frame_counter = 0
            self.lane_switching = 0

        return frame

    def yellowAndwhiteMask(self, frame):
        """
        Creates a mask for yellow and white colors in the frame
        :param frame: the current frame
        :return: the masked frame
        """
        yellow_th = [np.array([15, 50, 90]), np.array([25, 255, 255])]
        sens = 40
        white_th = [np.array([0, 0, 255 - sens]), np.array([255, sens, 255])]

        yellow_mask = cv.inRange(frame, yellow_th[0], yellow_th[1])
        white_mask = cv.inRange(frame, white_th[0], white_th[1])
        white_mask = cv.dilate(white_mask, np.ones((5, 2), dtype=np.uint8))
        return cv.bitwise_or(yellow_mask, white_mask)

    def preprocess(self, frame):
        """
        Preprocesses the frame by cropping it, creating a mask for yellow and white colors, removing noise and detecting edges
        :param frame: the current frame
        :return: the preprocessed frame
        """
        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # get the relevant crop of the frame
        trapeze_frame = self.createMask(frame_HSV)

        # convert the frame to grayscale
        trapeze_frame = cv.cvtColor(cv.cvtColor(trapeze_frame, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)

        # create a mask for yellow and white colors
        masked_frame = cv.bitwise_and(trapeze_frame, self.yellowAndwhiteMask(frame_HSV))

        # remove noise from the frame
        masked_frame = cv.GaussianBlur(masked_frame, (7, 7), cv.BORDER_WRAP)

        # detect edges in the frame
        return cv.Canny(masked_frame, 50, 150)

    def detect(self, processed_frame, org_frame):
        """
        Detects the lanes in the processed frame and draws them on the original frame, and checks for lane switching
        :param processed_frame: the preprocessed frame
        :param org_frame: the original frame
        :return: the original frame with the lanes drawn and any lane switching message
        """
        self.frame_processed += 1
        # get all lines in frame
        lines = cv.HoughLinesP(processed_frame, rho=3, theta=np.pi / 160, threshold=40, minLineLength=130, maxLineGap=180)
        # Draw the lanes on the original frame
        frame_with_lanes = self.drawLane(lines, org_frame)
        # Check for lane switching and add message to frame if necessary
        return self.laneSwitchDetection(frame_with_lanes)

    def chooseTwoLines(self, lines):
        if lines is None:
            return self.prev_lines_coords

        # Store the coordinates of the right and left lane lines
        right_x_lower = [0]
        right_x_upper = [0]
        left_x_lower = [0]
        left_x_upper = [0]

        # Absolute upper and lower border of the lane line we wish to draw
        y_lower = 625
        y_upper = 415

        # Iterate over all lines in the frame
        for line in lines:
            x1, y1, x2, y2 = line[0]

            line_angle = slope(x1, y1, x2, y2, self.frame_processed)
            b = y1 - line_angle * x1

            # Check the angle of the line, and append the coordinates to the appropriate lists
            if self.lane_switching == 0:
                if 0.33 < line_angle < 1.73:
                    right_x_lower.append((y_lower - b) / line_angle)
                    right_x_upper.append((y_upper - b) / line_angle)
                elif -1. < line_angle < -0.45:
                    left_x_lower.append((y_lower - b) / line_angle)
                    left_x_upper.append((y_upper - b) / line_angle)

            elif self.lane_switching == LEFT:
                if 0.33 < line_angle < 1.73:
                    right_x_lower.append((y_lower - b) / line_angle)
                    right_x_upper.append((y_upper - b) / line_angle)
                elif -1. < line_angle < -0.15:
                    left_x_lower.append((y_lower - b) / line_angle)
                    left_x_upper.append((y_upper - b) / line_angle)

            else:
                if 0.15 < line_angle < 1:
                    right_x_lower.append((y_lower - b) / line_angle)
                    right_x_upper.append((y_upper - b) / line_angle)
                elif -1. < line_angle < -0.45:
                    left_x_lower.append((y_lower - b) / line_angle)
                    left_x_upper.append((y_upper - b) / line_angle)

        # Calculate the median coordinates of the right and left lane lines
        right_low = self.createSafeCoords(np.median(right_x_lower), y_lower, 0)
        right_up = self.createSafeCoords(np.median(right_x_upper), y_upper, 1)
        left_low = self.createSafeCoords(np.median(left_x_lower), y_lower, 2)
        left_up = self.createSafeCoords(np.median(left_x_upper), y_upper, 3)

        return [right_low, right_up, left_low, left_up]

    def createSafeCoords(self, xval, yval, idx):
        """
        Create safe coordinates for the lane lines.
        This function is used to ensure that the lane line coordinates are not too far from the previous frame's coordinates.
        :param xval: The x value of the current lane line.
        :param yval: The y value of the current lane line.
        :param idx: The index of the current lane line.
        :return: A tuple containing the safe x and y coordinates of the lane line.
        """
        # check if the x value is less than 1 or if the difference between the current x value and the previous x value is greater than 150
        if xval < 1 or (np.abs(xval - self.prev_lines_coords[idx][0]) > 150):
            return self.prev_lines_coords[idx]
        return (int(xval), yval)


# Create an instance of LaneDetector class
LD = LaneDetector()

# Open a video stream
stream = cv.VideoCapture(getcwd() + "\\cropped_sample2.mp4")

# Read the first frame from the video stream
ret, frame = stream.read()

# Create a video writer object to save the output video
out = cv.VideoWriter('when I switch lanes.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                     (frame.shape[1], frame.shape[0]))

# Check if the video stream was opened successfully
if stream.isOpened() == False:
    print('Couldn\'t load file "sample.mp4"')
    exit(-1)

# Loop through the video frames
while stream.isOpened():
    # Read the next frame from the video stream
    ret, frame = stream.read()

    # Check if the frame was read successfully
    if ret:
        # Preprocess the frame
        processed_frame = LD.preprocess(frame)
        # Detect lanes in the frame
        drawn_frame = LD.detect(processed_frame, frame)
        # Show the processed frame
        cv.imshow('tmuna', drawn_frame)
        # Wait for a key press
        cv.waitKey(5)
        # Write the processed frame to the output video
        out.write(drawn_frame)
    else:
        break

# Release the video stream and close the output video
stream.release()
out.release()
# Close all open windows
cv.destroyAllWindows()
