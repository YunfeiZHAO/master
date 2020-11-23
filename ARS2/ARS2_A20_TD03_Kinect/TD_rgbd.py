#!/usr/bin/env python
import numpy as np
import cv2
import pptk


# pip install pptk # !! Ubuntu users, pptk has a bug with libz.so.1, see how to do a new symlink in
# https://github.com/heremaps/pptk/issues/3 from write_and_read_sync_imgs import open_shot Note: Be careful,
# "depths" given by Kinect are not actually "depths" but disparities. In common language, these terms are often
# considered the same.

# %%
def open_shot(n, fromdir='./shots'):
    # open a shot according to its number
    depth = cv2.imread('{}/{:02d}_depth.png'.format(fromdir, n), cv2.IMREAD_ANYDEPTH)
    depth_pretty = cv2.imread('{}/{:02d}_depth_pretty.png'.format(fromdir, n))
    bgr = cv2.imread('{}/{:02d}_rgb.png'.format(fromdir, n))
    return depth, depth_pretty, bgr


# %%
def raw_depth_to_meters(raw_disp):
    # get distances in meter from depth given by Kinect
    # Kinect computes the depth, but outputs a value coded a specific way in [[0;2047]] on 11 bits
    m_depth = np.zeros_like(raw_disp, dtype=np.float)
    # raw_disp must be transformed, some explanations in https://wiki.ros.org/kinect_calibration/technical

    # TODO
    b = 0.075  # m
    f = 580  # px
    doff = 1090  # 8 times px
    m_depth = b * f/(1/8 * (doff - raw_disp))

    return m_depth


# %%
def xyz_from_depth_meters(m_depth):
    # computes points 3D coordinates from depth
    xyz = np.zeros((m_depth.shape[0], m_depth.shape[1], 3), dtype=np.float)
    # TODO
    cx = 640/2
    cy = 480/2
    f = 580
    ind = np.indices(m_depth.shape)
    xyz[:, :, 2] = m_depth
    xyz[:, :, 0] = (ind[1] - cx) / f * m_depth
    xyz[:, :, 1] = (ind[0] - cy) / f * m_depth
    xyz = np.reshape(xyz, (-1, 3))

    return xyz


# %%
def color_of_xyz(xyz, rgb_img):
    # associate colors to depth / 3D points
    rgb_colors = np.reshape(rgb_img, (-1, 3))
    rgb_colors = rgb_colors[:, ::-1]  # ncv gbr
    # TODO
    # 1. translate from IR camera frame to RGB camera frame
    xyz_rgb_frame = np.copy(xyz)
    xyz_rgb_frame[:, 0] += 0.025

    # 2. back project xyz points into RGB camera image
    cx = 640/2
    cy = 480/2
    f = 525
    K = np.array([[f, 0, cx],
                 [0, f, cy],
                 [0, 0, 1]])
    xyz_cam = np.zeros_like(xyz)
    xyz_cam = K * xyz # find how to compute each triplet xyz
    # x and gives the position in the image
    # fill rgb_colors
    return rgb_colors


# %%
def main(shot):
    depth, depth_pretty, bgr = open_shot(shot)  # open RGB and D data of the shot
    # cv2.imshow('Depth', depth_pretty)  # display depth
    # cv2.imshow('RGB', bgr)  # display RGB

    # depth map in meters
    m_depth = raw_depth_to_meters(depth)

    # show in 3D point cloud
    xyz = xyz_from_depth_meters(m_depth)
    print(xyz)
    print(xyz.shape)

    # no-colored version
    v = pptk.viewer(xyz) # show 3D point cloud
    v.set(point_size=0.001)

    # add color information
    # rgb_colors = color_of_xyz(xyz, bgr)
    # v.attributes(rgb_colors / 255.)

    while 1:
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            # v.close()
            break

    # print(np.max(m_depth))
    # print(np.min(m_depth))


# %%
if __name__ == "__main__":
    # cv2.namedWindow('Depth')
    # cv2.namedWindow('RGB')
    # print('Press ESC in window to stop')

    shot = 1  # shot number to open
    main(shot)
