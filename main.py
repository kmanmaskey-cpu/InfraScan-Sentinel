# ============================================
# InfraScan Sentinel - Current State
# What it does: Fuses Hough + MiDaS edges adaptively
# Key outputs: fused_left, fused_right, real_gap_cm, r_squared
# Completed:
#   - Fused visualization ✓
#   - Confidence score display ✓
#   - R² clamping ✓
#   - Siding deduplication ✓
#   - EXIF focal length extraction ✓
#   - Physics-based scale formula ✓
#   - Batch processing on multiple images ✓
#   - Tested on 20-30 real Nepali images ✓
# Known issues:
#   - Assumed distance fixed at 500cm (needs ground truth)
# Next task: Collect ground truth data at Swayambhu
#            with tape measure + proper building gap photos
# ============================================
import cv2
import numpy as np
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from exif import get_focal_length_pixels
import glob
import pandas as pd


image_paths = glob.glob('C:\\ML PROJECTS\\InfraScan-Sentinel\\dataset\\*.jpg')

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform


def min_gap(h1,h2):
    seismic_gap=((0.025*h1)+(0.025*h2))*100
    return seismic_gap


def process_image(image_path):
    inner_left_edge =None
    inner_right_edge = None
    gap_data = []
    valid_gaps = []
    gaps = []
    horizontal_y_positions = []
    h1=12    #HARDCODED#
    h2=9   



    # 1. Load your building photo
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load: {image_path}")
        return
    # 2. Convert to Grayscale
    RGB = cv2.cvtColor(image,cv2.COLOR_BGRA2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    input_batch = transform(RGB).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction=prediction.squeeze()


    depth_map = prediction.cpu().numpy()
    print(depth_map.shape[0])
    h = depth_map.shape[0]
    middle_row = depth_map[h//2]


    x_scale = image.shape[0]/depth_map.shape[0]
    y_scale = image.shape[1]/depth_map.shape[1]

    depth_gradient = np.abs(np.gradient(middle_row))

    threshold = np.percentile(depth_gradient, 90)  # top 10% strongest edges
    edge_pixels_depth = np.where(depth_gradient > threshold)[0]

    edge_pixels_original = edge_pixels_depth * x_scale

    depth_left_edge = edge_pixels_original[
        edge_pixels_original < image.shape[1]//2]
    depth_right_edge = edge_pixels_original[
        edge_pixels_original > image.shape[1]//2]
    
  
    
        

    if len(depth_left_edge) == 0 or len(depth_right_edge) == 0:
        print(f"Skipping {image_path} - depth edges not found on both sides")
        return  # exit function early, move to next image
    else:
        depth_left_edge=depth_left_edge.max()
        depth_right_edge=depth_right_edge.min()
        left_idx = int(depth_left_edge / x_scale)
        right_idx = int(depth_right_edge / x_scale)
        gap_sharpness = np.mean(depth_gradient[left_idx:right_idx])
        scale_factor = 2 #hardcoded
        gap_sharpness_normalized = 1 / (1 + np.exp(-gap_sharpness/ scale_factor))
        print('The average sharpness of the image is ',gap_sharpness_normalized)



    # 3. Use Canny Edge Detection
    # This finds the "lines" between the buildings
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 
        rho=1,            # Distance resolution in pixels (usually 1)
        theta=np.pi/180,  # Angle resolution in radians (1 degree)
        threshold=50,  # Minimum 'votes' to be considered a line
        minLineLength=60,# Minimum length of line in pixels
        maxLineGap=60 # Max gap between points to link them
    )

    left = []
    right = []
    
    mid = image.shape[1]//2

    cm_per_pixel = 1.0  # Default: 1 pixel = 1 cm


    # 3. Draw the lines back onto the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 1. Calculate the angle of the line
            # arctan2 returns radians, we convert to degrees
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            # 2. The Vertical Filter 
            # We only want lines between 70 and 110 degrees
            if 75 < angle < 105:
                
                if x1<mid:
                    left.append(x1)
                else:
                    right.append(x1)
            # 2. Capture Horizontal lines (The Scale Vectors)
            # We look for lines near 0 or 180 degrees
            elif angle < 10 or angle > 170:
            
                horizontal_y_positions.append(y1)





    if left:
        
        ignore_zone = image.shape[1]*0.1  # Ignore the leftmost 20% of the image to avoid false positives from the edge
        left = [x for x in left if x>ignore_zone]  # Filter out lines in the ignore zone
        
        counts, bin_edges = np.histogram(left, bins=10)
        best_bin_index = np.argmax(counts)

        
        # Define the boundaries of our "Busiest Block"
        lower_bound = bin_edges[best_bin_index]
        upper_bound = bin_edges[best_bin_index + 1]

        # Filter the left edges to only include those within the busiest block
        
        refined_left = [x for x in left if lower_bound <= x <= upper_bound]
        if not refined_left:
            pass
        else:

            # Now find the final edge
            inner_left_edge = max(refined_left)

    if right:
        counts, bin_edges = np.histogram(right, bins=10)
        best_bin_index = np.argmax(counts)
        
        # Define the boundaries of our "Busiest Block"
        lower_bound = bin_edges[best_bin_index]
        upper_bound = bin_edges[best_bin_index + 1]

        # Filter the right edges to only include those within the busiest block
        
        refined_right = [x for x in right if lower_bound <= x <= upper_bound]
        if not refined_right:
            pass
        else:

            # Now find the final edge
            inner_right_edge = min(refined_right)

    # Find the average pixel distance between siding boards
    if len(horizontal_y_positions) >= 2:
        horizontal_y_positions.sort()
        deduped = [horizontal_y_positions[0]]
        for y in horizontal_y_positions[1:]:
            if y - deduped[-1] > 3:
                deduped.append(y)
        horizontal_y_positions = deduped
        # Calculate differences between consecutive lines
        gaps = np.diff(horizontal_y_positions)
        print(f"Gap range: min={min(gaps):.1f}, max={max(gaps):.1f}, mean={np.mean(gaps):.1f}")
        
        # Filter out tiny gaps (noise) and huge gaps (missed boards)
        # Most siding boards in pixels will be roughly consistent
        valid_gaps = [g for g in gaps if 2 < g < 50]
        for i in range(len(valid_gaps)):
            y_pos = horizontal_y_positions[i]
            if 2 < gaps[i] < 50:
                gap_data.append((y_pos, gaps[i]))

        if len(gap_data) > 2:  # We need at least a few points to find a 'trend'
            # 1. Convert our list of tuples into two separate math arrays
            y_coords = np.array([pt[0] for pt in gap_data])
            pixel_gaps = np.array([pt[1] for pt in gap_data])
            
            # 2. Find the OPTIMAL slope (m) and intercept (c)
            m, c = np.polyfit(y_coords, pixel_gaps, 1)
            
            # 3. Predict what the pixel gap "should be" at the vertical center
            # where our building measurement is happening
            y_target = image.shape[0] // 2
            optimal_pixel_gap = m * y_target + c

            total_residual = np.sum((pixel_gaps-optimal_pixel_gap)**2)
            variance = np.sum((pixel_gaps - np.mean(pixel_gaps))**2)
            r_squared = 1 - (total_residual / (variance + 1e-7))# Add small value to prevent division by zero
            r_squared = max(0.0, r_squared)  
            print(f"R² of the fit: {r_squared:.4f}")
            
            # 4. Use this refined pixel size for the final math
            known_siding_cm = 10.16
            cm_per_pixel = known_siding_cm / (optimal_pixel_gap + 1e-7)  #to be used in calibration ayer


        

        
        elif valid_gaps:
            avg_pixel_gap = np.mean(valid_gaps)
            
            # ENGINEERING CONSTANT: 
            # Standard "Lap Siding" is usually 4 inches or 10.16 cm exposure
            known_siding_cm = 10.16 
            
            # The Magic Formula: Scale = Real World / Pixels
            cm_per_pixel = known_siding_cm / (avg_pixel_gap + 1e-7)  # Add small value to prevent division by zero
            print(f"AUTOMATED SCALE: {cm_per_pixel:.4f} cm/pixel")



  
    if inner_left_edge is not None and inner_right_edge is not None:
        # Check if we successfully found BOTH edges using our refined logic
        """h1 = int(input('enter the height of building 1'))
        h2 = int(input('enter the height of building 2'))"""
    
        w_hough = 1- gap_sharpness_normalized
        
        fused_left = (inner_left_edge * w_hough) + (depth_left_edge * gap_sharpness_normalized)
        fused_right = (inner_right_edge * w_hough) + (depth_right_edge * gap_sharpness_normalized)
        
        print(f"Hough weight: {w_hough:.2f}, gap_sharpness_normalizedt: {gap_sharpness_normalized:.2f}")
        print(f"Raw gap_sharpness: {gap_sharpness:.4f}")
        print(f"Fused left edge: {fused_left:.1f}")
        print(f"Fused right edge: {fused_right:.1f}")
        
        # 1. THE MATH: Use the variables created by your histogram blocks
        pixel_gap = fused_right - fused_left
        
        assumed_distance_cm = 500
        try:
            focal_length_px = get_focal_length_pixels(image_path)
        except:
            focal_length_px = 2500  # reasonable smartphone default
        real_world_gap = (pixel_gap * assumed_distance_cm) / focal_length_px
        if real_world_gap <min_gap(h1,h2):  # If the gap is less than 10 cm, we consider it a "collision risk"
            status = "WARNING: Collision Risk Detected!"
            color = (0, 0, 255)  # Red
        else:
            status = "Gap is safe."
            color = (0, 255, 0)  # Green if safe
        
        # 2. THE DRAWING: Visualizing the measurement
        y_mid = image.shape[0] // 2
        
        
        
        # 1. Draw the dynamic bridge (Red if dangerous, Green if safe)
        print(f"Hough left edge: {inner_left_edge}")
        cv2.line(image, (int(fused_left), y_mid), (int(fused_right), y_mid), color, 5)
        
        # 2. Show the primary measurement (CM) at the top
        cv2.putText(image, f"GAP: {real_world_gap:.1f}cm", (int(fused_left), y_mid - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  

        # 3. Show the Status Verdict slightly below it
        cv2.putText(image, status, (int(fused_left), y_mid - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        confidence = int(gap_sharpness_normalized * 100)
        cv2.putText(image, f"CONFIDENCE: {confidence}% (sharpness={gap_sharpness_normalized:.2f})", 
                (int(fused_left), y_mid - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        print(f"REFINED Detected Gap: {pixel_gap:.2f} pixels")
        print(f"REFINED Estimated Real-World Gap: {real_world_gap:.2f} cm")
        cv2.imshow('InfraScan Sentinel - Seismic Audit', image)
        print(f"Gap data points found: {len(gap_data)}")
        print(f"Horizontal lines found: {len(horizontal_y_positions)}")
        print(f"Valid gaps found: {len(valid_gaps)}")
            

    output_path = image_path.replace('.jpg', '_result.jpg')
    cv2.imwrite(output_path, image)
    print(f"Saved result to: {output_path}")

for path in image_paths:
    process_image(path)



