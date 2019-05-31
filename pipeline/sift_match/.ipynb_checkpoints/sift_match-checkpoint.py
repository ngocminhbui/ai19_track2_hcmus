import os
import glob
import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [60, 60]

def read_features_file(filename):
    features = []
    keypoints = []
    with open(filename,'r') as f:
        s = [i.strip() for i in f.readlines()]
    m = int(s[1])

    for i in range(0, m):
        sh = s[2+i].split()
        j = [float(k) for k in sh[5:]]
        pt =  (float(sh[0]), float(sh[1])) 
        kp = cv.KeyPoint(x=pt[0], y=pt[1], _size=0)
        features.append(j)
        keypoints.append(kp)
    features = np.array(features, dtype=np.float32)
    return keypoints, features
        
def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv.line(new_img, end1, end2, c, thickness)
        cv.circle(new_img, end1, r, c, thickness)
        cv.circle(new_img, end2, r, c, thickness)
    
    plt.figure(figsize=(15,15))
    plt.imshow(new_img)
    plt.show()
    

def do_match(a_feature_filename, b_feature_filename, a_image_filename, b_image_filename, show_result = True):
    cv.setRNGSeed(0)
    img_object = cv.imread(a_image_filename, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(b_image_filename, cv.IMREAD_GRAYSCALE)
    
    if img_object is None or img_scene is None:
        if show_result:
            print('Could not open or find the images!')
            return
        else:
            return False
    
    keypoints_obj, descriptors_obj = read_features_file(a_feature_filename)
    keypoints_scene, descriptors_scene = read_features_file(b_feature_filename)

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.9
    good_matches = []
    for m,n in knn_matches:
        if m.distance < 0.5:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    selected_matches = []

    if len(good_matches) == 0:
        if show_result:
            plt.imshow(img_scene)
            print('No good matches found!')
            return
        else:
            return False
    print(len(good_matches), '/', len(knn_matches))
    
    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)

    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
        
    H, _ =  cv.findHomography(obj, scene, cv.RANSAC, 25)
    
#     print(len(_))
#     print(len(good_matches))
#     print(_)
    
    for i in range(len(_)):
        if _[i] == 1:
            selected_matches.append(good_matches[i])
#     print(selected_matches)
#     print(np.sum(_==1))

    if H is None:
        if show_result:
            plt.imshow(img_scene)
            print('Can''t find homography!')
            return
        else:
            return False
    
    if show_result:
        #-- Draw matches
        img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
#         cv.drawMatches(img_object, 
#                        keypoints_obj, 
#                        img_scene, 
#                        keypoints_scene, 
#                        good_matches, 
#                        (255,255,0)
#                        )
        
        cv.drawMatches(img_object, 
                       keypoints_obj, 
                       img_scene, 
                       keypoints_scene, 
                       selected_matches, 
                       img_matches, 
                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)        
#         cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, matchColor=(255, 255, 0), flags=2)
        #-- Get the corners from the image_1 ( the object to be "detected" )
        obj_corners = np.empty((4,1,2), dtype=np.float32)
        obj_corners[0,0,0] = 0
        obj_corners[0,0,1] = 0
        obj_corners[1,0,0] = img_object.shape[1] # shape 1 is width, shape 0 is height
        obj_corners[1,0,1] = 0
        obj_corners[2,0,0] = img_object.shape[1]
        obj_corners[2,0,1] = img_object.shape[0]
        obj_corners[3,0,0] = 0
        obj_corners[3,0,1] = img_object.shape[0]

        scene_corners = cv.perspectiveTransform(obj_corners, H)
        #-- Draw lines between the corners (the mapped object in the scene - image_2 )
        cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
            (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
        cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
            (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
        cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
            (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
        cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
            (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
        #-- Show detected matches
        plt.imshow(img_matches)
    else:
        return len(good_matches)

def is_match(a,b,show_result = True):
    return do_match(a_image_filename=a[0],
             a_feature_filename=a[1],
             b_image_filename=b[0],
             b_feature_filename=b[1],
             show_result=show_result)

IMG_TEM = "/home/hthieu/AICityChallenge2019/data/Track2Data/{}"
FEA_TEM = "/home/tmkhiem/contest/AICityChallenge2019/data/root_sift_feat/{}.hesaff.sift"

def compare(img1, img2, show_result = False):
    return do_match(
        a_image_filename=IMG_TEM.format(img1),
        a_feature_filename=FEA_TEM.format(img1),
        b_image_filename=IMG_TEM.format(img2),
        b_feature_filename=FEA_TEM.format(img2),
        show_result=show_result)

def visualize_keypoints(img_path):
    keypoints_obj, descriptors_obj = read_features_file(
        FEA_TEM.format(img_path))
    img = cv.imread(
       IMG_TEM.format(img_path), cv.IMREAD_GRAYSCALE)
    
    plt.imshow(cv.drawKeypoints(img, keypoints_obj, None))