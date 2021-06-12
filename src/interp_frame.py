import numpy as np
import cv2 as cv

def write_images(img0, img1, flow):
    cv.imwrite("img0.png", img0)
    cv.imwrite("img1.png", img1)
    cv.imwrite("flow.png", flow)

def flow_to_color(flow, hsv):
    # brighter pixels will correspond to higher flows, and the color will correspond to the direction
    mag, angle = cv.cartToPolar(flow[..., 0], flow[..., 1]) # Computes the magnitude and angle of the 2D vectors
    hsv[..., 0] = angle*180/np.pi/2 # Sets image hue according to the optical flow direction
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX) # Sets image value according to the optical flow magnitude (normalized)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR) # Converts HSV to RGB (BGR) color representation

def forward_warping(src, dst, t, flow):
    # prev_frame[y, x] == curr_frame[y + flow[y, x, 1], x + flow[y, x, 0]]
    h, w, ch = src.shape
    mid_frame = np.zeros_like(src)
    x_src = np.arange(w)
    y_src = np.arange(h)
    xp_src, yp_src = np.meshgrid(x_src,y_src)
    src_cood = np.vstack(list(map(np.ravel, [yp_src, xp_src]))) # image is (row, column) indexing

    flow_vectors = flow.ravel().reshape(-1, 2).T
    flow_vectors[[0,1]] = flow_vectors[[1,0]] # flow is (x,y) coordinate-based, which is (column, row) indexing

    dst_cood = (np.round(src_cood + t * flow_vectors)).astype(int) # (2, 226592)
    dst_cood[0] = np.clip(dst_cood[0], 0, h-1)
    dst_cood[1] = np.clip(dst_cood[1], 0, w-1)
    mid_frame[dst_cood[0], dst_cood[1]] = src[src_cood[0], src_cood[1]]

    return mid_frame
def optical_flow(img0, img1):
    gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY) # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) 
    hsv = np.zeros_like(img0) #zero intensities with same type as img0 (388, 584, 3) 
    hsv[:,:,1] = 255 # Sets image saturation to maximum 
    
    flow = cv.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0) # (384, 584, 2) dense optical flow for all pixels
    # rgb_flow = flow_to_color(flow, hsv) # show flow image, nothing to do with interpolated frame
    # write_images(img0, img1, rgb_flow)
    # gray0 = gray1 # update previous frame
    return flow

def splatting(frame0, frame1):
    return (frame0 + frame1)
def interp_frame(img0, img1, t):
    flow0_1 = optical_flow(img0, img1)
    mid_frame0_1 = forward_warping(img0, img1, t, flow0_1)
    flow1_0 = optical_flow(img0, img1)
    mid_frame1_0 = forward_warping(img0, img1, 1-t, flow1_0)
    # mid_frame = splatting(mid_frame0_1, mid_frame1_0)
    mid_frame = mid_frame0_1
    # a = np.array([1,1,3,0, 0, 0,4,5, 0,0,2,4]).reshape(2,2, 3)
    # mask = a[:,:,0] == 0 and a[:,:,1] == 0 and a[:,:,2] == 0
    # print(a)
    # mask = a[:,:,:] = np.array([0,0,0])
    # print(a)
    # b = a[:, :, mask]
    # b = np.argwhere(a[:,:,:] == np.array([0,0,0]))
    # print(b)
    return mid_frame


    