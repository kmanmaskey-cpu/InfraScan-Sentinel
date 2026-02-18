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
mid = image.shape[1]//2
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



# Check if we successfully found BOTH edges using our refined logic
if 'inner_left_edge' in locals() and 'inner_right_edge' in locals():
    
    # 1. THE MATH: Use the variables created by your histogram blocks
    pixel_gap = inner_right_edge - inner_left_edge
    cm_per_pixel = 0.06  # This is a made-up conversion factor; you'd need to calibrate this for your specific image
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