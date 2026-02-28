import cv2
import numpy as np

# 1. Load your building photo
image = cv2.imread('C:\\ML PROJECTS\\InfraScan-Sentinel\\OIP.jpg')
# 2. Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
horizontal_y_positions = []
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
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if x1<mid:
                left.append(x1)
            else:
                right.append(x1)
        # 2. Capture Horizontal lines (The Scale Vectors)
        # We look for lines near 0 or 180 degrees
        elif angle < 10 or angle > 170:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue lines
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

    # Now find the final edge
    inner_right_edge = min(refined_right)

# Find the average pixel distance between siding boards
if len(horizontal_y_positions) >= 2:
    horizontal_y_positions.sort()
    # Calculate differences between consecutive lines
    gaps = np.diff(horizontal_y_positions)

    gap_data=[]




    # Filter out tiny gaps (noise) and huge gaps (missed boards)
    # Most siding boards in pixels will be roughly consistent
    valid_gaps = [g for g in gaps if 5 < g < 50]
    for i in range(len(valid_gaps)):
        y_pos = horizontal_y_positions[i]
        if 5 < gaps[i] < 50:
            gap_data.append((y_pos, gaps[i]))

    if len(gap_data) > 3:  # We need at least a few points to find a 'trend'
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
        r_squared = 1 - (total_residual / (variance + 1e-7))  # Add small value to prevent division by zero
        print(f"R² of the fit: {r_squared:.4f}")
        
        # 4. Use this refined pixel size for the final math
        known_siding_cm = 10.16
        cm_per_pixel = known_siding_cm / (optimal_pixel_gap + 1e-7)

    

    
    elif valid_gaps:
        avg_pixel_gap = np.mean(valid_gaps)
        
        # ENGINEERING CONSTANT: 
        # Standard "Lap Siding" is usually 4 inches or 10.16 cm exposure
        known_siding_cm = 10.16 
        
        # The Magic Formula: Scale = Real World / Pixels
        cm_per_pixel = known_siding_cm / (avg_pixel_gap + 1e-7)  # Add small value to prevent division by zero
        print(f"AUTOMATED SCALE: {cm_per_pixel:.4f} cm/pixel")



# Check if we successfully found BOTH edges using our refined logic
if 'inner_left_edge' in locals() and 'inner_right_edge' in locals():
    
    # 1. THE MATH: Use the variables created by your histogram blocks
    pixel_gap = inner_right_edge - inner_left_edge
    real_world_gap = pixel_gap*cm_per_pixel
    if real_world_gap <4:  # If the gap is less than 10 cm, we consider it a "collision risk"
        status = "WARNING: Collision Risk Detected!"
        color = (0, 0, 255)  # Red
    else:
        status = "Gap is safe."
        color = (0, 255, 0)  # Green if safe
    
    # 2. THE DRAWING: Visualizing the measurement
    y_mid = image.shape[0] // 2
    
    
    
    # 1. Draw the dynamic bridge (Red if dangerous, Green if safe)
    print(f"Hough left edge: {inner_left_edge}")
    print(f"Hough right edge: {inner_right_edge}")
    cv2.line(image, (int(inner_left_edge), y_mid), (int(inner_right_edge), y_mid), color, 5)
    
    # 2. Show the primary measurement (CM) at the top
    cv2.putText(image, f"GAP: {real_world_gap:.1f}cm", (int(inner_left_edge), y_mid - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  

    # 3. Show the Status Verdict slightly below it
    cv2.putText(image, status, (int(inner_left_edge), y_mid - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    print(f"REFINED Detected Gap: {pixel_gap:.2f} pixels")
    print(f"REFINED Estimated Real-World Gap: {real_world_gap:.2f} cm")
    cv2.imshow('InfraScan Sentinel - Seismic Audit', image)


if cv2.waitKey(0) & 0xFF == ord('q'): 
    pass

cv2.destroyAllWindows()