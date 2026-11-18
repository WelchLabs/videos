from welch_paths import welch_path
import manimgx as m
from welch_scene import WelchScene, image_at_height
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from functools import partial

 
CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'

graphics_dir=welch_path('backprop_3/graphics/')

def create_3d_polygon_regions(polygons, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.3):
    """
    Create 3D polygons by mapping each corner point to its second layer output height.
    
    Args:
        polygons: List of 2D polygon corner points from simple_polygon_finder
        w1, b1: First layer weights and biases
        w2, b2: Second layer weights and biases  
        neuron_idx: Which second layer neuron to visualize
        viz_scale: Scaling factor for z-coordinate
    
    Returns:
        List of 3D polygon objects
    """
    
    def evaluate_second_layer_at_point(x, y):
        """Evaluate the second layer neuron output at a specific (x,y) point"""
        # First layer outputs
        linear_1 = w1[0,0] * x + w1[0,1] * y + b1[0]
        relu_1 = max(0, linear_1)
        
        linear_2 = w1[1,0] * x + w1[1,1] * y + b1[1]
        relu_2 = max(0, linear_2)
        
        # Second layer output (no ReLU applied here to see full surface)
        second_layer_output = w2[neuron_idx,0] * relu_1 + w2[neuron_idx,1] * relu_2 + b2[neuron_idx]
        
        return second_layer_output * viz_scale
    
    polygon_objects = []
    colors = [m.RED, BLUE, GREEN, YELLOW, m.PURPLE, m.ORANGE]
    
    for i, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
            
        # Map each 2D corner point to 3D
        points_3d = []
        for point_2d in polygon:
            x, y = point_2d
            z = evaluate_second_layer_at_point(x, y)
            points_3d.append([x, y, z])
        
        # Create the 3D polygon
        color = colors[i % len(colors)]
        poly_3d = m.Polygon(*points_3d,
                         fill_color=color,
                         fill_opacity=0.7,
                         stroke_color=color,
                         stroke_width=2)
        
        polygon_objects.append(poly_3d)
    
    return polygon_objects


def create_3d_polygon_regions_with_relu(polygons, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.3):
    """
    Creates 3D polygons with ReLU applied, splitting polygons that cross z=0.
    """
    
    def evaluate_second_layer_linear(x, y):
        """Evaluate the second layer neuron LINEAR output (before ReLU)"""
        # First layer outputs
        linear_1 = w1[0,0] * x + w1[0,1] * y + b1[0]
        relu_1 = max(0, linear_1)
        
        linear_2 = w1[1,0] * x + w1[1,1] * y + b1[1]
        relu_2 = max(0, linear_2)
        
        # Second layer LINEAR output (before ReLU)
        second_layer_linear = w2[neuron_idx,0] * relu_1 + w2[neuron_idx,1] * relu_2 + b2[neuron_idx]
        
        return second_layer_linear * viz_scale
    
    def find_zero_crossing_point(p1, p2, z1, z2):
        """Find the point where a line segment crosses z=0"""
        if abs(z2 - z1) < 1e-10:
            return None  # Line is parallel to z=0
        
        # Linear interpolation to find where z=0
        t = -z1 / (z2 - z1)
        if 0 <= t <= 1:
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            return [x, y, 0]
        return None
    
    def split_polygon_at_zero(polygon_2d):
        """Split a polygon into parts above and below z=0"""
        # Evaluate z at each corner
        corners_3d = []
        z_values = []
        
        for point_2d in polygon_2d:
            x, y = point_2d
            z_linear = evaluate_second_layer_linear(x, y)
            corners_3d.append([x, y, z_linear])
            z_values.append(z_linear)
        
        # Check if polygon crosses z=0
        has_positive = any(z > 1e-10 for z in z_values)
        has_negative = any(z < -1e-10 for z in z_values)
        
        if not has_negative:
            # Entire polygon is above z=0, apply ReLU normally
            points_3d_relu = [[p[0], p[1], max(0, p[2])] for p in corners_3d]
            return [points_3d_relu], []
        
        elif not has_positive:
            # Entire polygon is below z=0, gets clipped to z=0
            points_2d_on_plane = [[p[0], p[1], 0] for p in corners_3d]
            return [], [points_2d_on_plane]
        
        else:
            # Polygon crosses z=0, need to split it
            above_points = []
            on_plane_points = []
            
            n = len(corners_3d)
            for i in range(n):
                current = corners_3d[i]
                next_point = corners_3d[(i + 1) % n]
                
                current_z = current[2]
                next_z = next_point[2]
                
                # Add current point if it's above z=0
                if current_z > 1e-10:
                    above_points.append([current[0], current[1], current_z])
                elif abs(current_z) <= 1e-10:
                    # On the plane
                    above_points.append([current[0], current[1], 0])
                    on_plane_points.append([current[0], current[1], 0])
                else:
                    # Below plane - add to on_plane_points
                    on_plane_points.append([current[0], current[1], 0])
                
                # Check for zero crossing on edge to next point
                if (current_z > 1e-10 and next_z < -1e-10) or (current_z < -1e-10 and next_z > 1e-10):
                    crossing_point = find_zero_crossing_point(current, next_point, current_z, next_z)
                    if crossing_point:
                        above_points.append(crossing_point)
                        on_plane_points.append(crossing_point)
            
            result_above = [above_points] if len(above_points) >= 3 else []
            result_on_plane = [on_plane_points] if len(on_plane_points) >= 3 else []
            
            return result_above, result_on_plane
    
    polygon_objects = []
    colors = [m.RED, BLUE, GREEN, YELLOW, m.PURPLE, m.ORANGE]
    
    color_count=0
    for i, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        
        # Split the polygon at z=0
        above_polygons, on_plane_polygons = split_polygon_at_zero(polygon)
        
        
        
        # Add polygons above z=0 (with their original heights)
        for poly_points in above_polygons:
            if len(poly_points) >= 3:
                color = colors[color_count % len(colors)]
                poly_3d = m.Polygon(*poly_points,
                                 fill_color=color,
                                 fill_opacity=0.7,
                                 stroke_color=color,
                                 stroke_width=2)
                polygon_objects.append(poly_3d)
                color_count+=1
        
        # Add polygons on the z=0 plane (clipped parts)
        for poly_points in on_plane_polygons:
            if len(poly_points) >= 3:
                color = colors[color_count % len(colors)]
                poly_flat = m.Polygon(*poly_points,
                                   fill_color=color,
                                   fill_opacity=0.4,  # More transparent for clipped parts
                                   stroke_color=color,
                                   stroke_width=2)
                polygon_objects.append(poly_flat)
                color_count+=1
    
    return polygon_objects



def get_polygon_corners(joint_points_1, joint_points_2, extent=1):
    """
    Compute the corner points of polygons formed by two ReLU joint lines
    and the boundaries of the [-extent, extent] × [-extent, extent] plane.
    
    Args:
        joint_points_1: List of [x, y] points defining first joint line
        joint_points_2: List of [x, y] points defining second joint line  
        extent: Boundary of the plane (default 1 for [-1,1] × [-1,1])
        
    Returns:
        List of polygons, where each polygon is a list of [x, y] corner points
    """
    import math
    
    def line_intersection(p1, p2, p3, p4):
        """Find intersection of two lines defined by points (p1,p2) and (p3,p4)"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
            
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        return [x, y]
    
    def point_on_boundary(point, extent):
        """Check if point is on the boundary of the square"""
        x, y = point
        return (abs(x - extent) < 1e-8 or abs(x + extent) < 1e-8 or 
                abs(y - extent) < 1e-8 or abs(y + extent) < 1e-8)
    
    def extend_line_to_boundary(p1, p2, extent):
        """Extend a line segment to the boundary of the square"""
        # Find intersections with all four boundaries
        boundaries = [
            [[-extent, -extent], [extent, -extent]],  # bottom
            [[extent, -extent], [extent, extent]],    # right  
            [[extent, extent], [-extent, extent]],    # top
            [[-extent, extent], [-extent, -extent]]   # left
        ]
        
        intersections = []
        for boundary in boundaries:
            intersection = line_intersection(p1, p2, boundary[0], boundary[1])
            if intersection is not None:
                x, y = intersection
                # Check if intersection is within boundary segment and square
                if (-extent <= x <= extent and -extent <= y <= extent):
                    intersections.append(intersection)
        
        return intersections
    
    # Collect all critical points
    critical_points = []
    
    # Add square corners
    corners = [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]]
    critical_points.extend(corners)
    
    # Process joint lines
    lines = []
    if joint_points_1 and len(joint_points_1) >= 2:
        lines.append(joint_points_1[:2])
    if joint_points_2 and len(joint_points_2) >= 2:
        lines.append(joint_points_2[:2])
    
    # Add intersections of joint lines with boundaries
    for line in lines:
        boundary_intersections = extend_line_to_boundary(line[0], line[1], extent)
        critical_points.extend(boundary_intersections)
    
    # Add intersection between the two joint lines (if they intersect)
    if len(lines) == 2:
        line_intersection_point = line_intersection(lines[0][0], lines[0][1], 
                                                  lines[1][0], lines[1][1])
        if (line_intersection_point is not None and 
            -extent <= line_intersection_point[0] <= extent and
            -extent <= line_intersection_point[1] <= extent):
            critical_points.append(line_intersection_point)
    
    # Remove duplicates
    unique_points = []
    for point in critical_points:
        is_duplicate = False
        for existing in unique_points:
            if (abs(point[0] - existing[0]) < 1e-8 and 
                abs(point[1] - existing[1]) < 1e-8):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)
    
    # Now identify which points belong to which polygon regions
    def get_region_id(point):
        """Determine which region a point belongs to based on line orientations"""
        x, y = point
        region_id = 0
        
        # Check which side of each line the point is on
        for i, line in enumerate(lines):
            x1, y1 = line[0]
            x2, y2 = line[1]
            
            # Cross product to determine side
            cross = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
            if cross > 1e-8:
                region_id |= (1 << i)  # Set bit i if on positive side
        
        return region_id
    
    # Group points by region
    regions = {}
    for point in unique_points:
        region_id = get_region_id(point)
        if region_id not in regions:
            regions[region_id] = []
        regions[region_id].append(point)
    
    # Convert regions to ordered polygons
    polygons = []
    for region_id, points in regions.items():
        if len(points) >= 3:
            # Calculate centroid
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            
            # Sort points by angle around centroid
            def angle_from_centroid(point):
                return math.atan2(point[1] - cy, point[0] - cx)
            
            ordered_points = sorted(points, key=angle_from_centroid)
            polygons.append(ordered_points)
    
    return polygons



def simple_polygon_finder(joint_points_1, joint_points_2, extent=1):
    """
    Simplified approach - manually construct the obvious polygons
    """
    if not joint_points_1 or not joint_points_2:
        return []
    
    line1 = joint_points_1[:2]
    line2 = joint_points_2[:2] 
    
    # Find intersection of the two lines
    def line_intersection(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2  
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
            
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        return [x, y]
    
    intersection = line_intersection(line1[0], line1[1], line2[0], line2[1])
    
    # For your specific case, let's manually build the polygons:
    polygons = []
    
    # Bottom triangle: intersection point, (0.41, -1), (1, -1), (1, 0.31)
    if intersection:
        poly1 = [intersection, [0.41, -1], [1, -1], [1, 0.31]]
        polygons.append(poly1)
        
        # Top right polygon: intersection, (1, 0.31), (1, 0.64)  
        poly2 = [intersection, [1, 0.31], [1, 0.64]]
        polygons.append(poly2)
        
        # Left polygon: intersection, (-1, 0.27), (-1, -1), (0.41, -1)
        poly3 = [intersection, [-1, 0.27], [-1, -1], [0.41, -1]]
        polygons.append(poly3)
        
        # Top polygon: intersection, (1, 0.64), (1, 1), (-1, 1), (-1, 0.27)
        poly4 = [intersection, [1, 0.64], [1, 1], [-1, 1], [-1, 0.27]]
        polygons.append(poly4)
    
    return polygons



def get_relu_joint(weight_1, weight_2, bias, extent=1):
    if m.np.abs(weight_2) < 1e-8:
        x_intercept = -bias / weight_1
        return [[x_intercept, -extent], [x_intercept, extent]] if -extent <= x_intercept <= extent else []
    elif m.np.abs(weight_1) < 1e-8:
        y_intercept = -bias / weight_2
        return [[-extent, y_intercept], [extent, y_intercept]] if -extent <= y_intercept <= extent else []
    else:
        points = []
        for x in [-extent, extent]:
            y = (-x * weight_1 - bias) / weight_2
            if -extent <= y <= extent: points.append([x, y])
        for y in [-extent, extent]:
            x = (-y * weight_2 - bias) / weight_1
            if -extent <= x <= extent: points.append([x, y])
        unique_points = []
        for p in points:
            is_duplicate = False
            for existing in unique_points:
                if abs(p[0] - existing[0]) < 1e-8 and abs(p[1] - existing[1]) < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(p)
        return unique_points

def line_from_joint_points_1(joint_points):
    if joint_points:
        # Create 3D points for the joint line
        joint_3d_points = []
        for point in joint_points:
            x, y = point
            z = 0
            joint_3d_points.append([x, y, z])
        
        if len(joint_3d_points) >= 2:
            joint_line = m.DashedLine(
                start=[joint_points[0][0], joint_points[0][1], 0],
                end=[joint_points[1][0], joint_points[1][1], 0],
                color=m.WHITE,
                stroke_width=3,
                dash_length=0.05
            )
            return joint_line


def surface_func_general(u, v, w1, w2, b, viz_scale=0.5):
    linear_output = w1 * u + w2 * v + b
    relu_output = max(0, linear_output)
    z = relu_output * viz_scale 
    return m.np.array([u, v, z])


def surface_func_second_layer(u, v, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.5):
    """
    Surface function for second layer neurons that combines first layer outputs.
    
    Args:
        u, v: Input coordinates (-1 to 1)
        w1: First layer weights (2x2 matrix)
        b1: First layer biases (2 element array)
        w2: Second layer weights (2x2 matrix) 
        b2: Second layer biases (2 element array)
        neuron_idx: Which second layer neuron (0 or 1)
        viz_scale: Scaling factor for visualization
    """
    
    # First layer neuron 1 output
    linear_output_1 = w1[0,0] * u + w1[0,1] * v + b1[0]
    relu_output_1 = max(0, linear_output_1)
    
    # First layer neuron 2 output  
    linear_output_2 = w1[1,0] * u + w1[1,1] * v + b1[1]
    relu_output_2 = max(0, linear_output_2)
    
    # Second layer neuron computation
    second_layer_input = w2[neuron_idx,0] * relu_output_1 + w2[neuron_idx,1] * relu_output_2 + b2[neuron_idx]
    second_layer_output = max(0, second_layer_input)
    
    # Use output as z-coordinate
    z = second_layer_output * viz_scale
    
    return m.np.array([u, v, z])

def surface_func_second_layer_no_relu_multi(u, v, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.5):
    """
    Surface function for second layer neurons that combines outputs from any number of first layer neurons.
    
    Args:
        u, v: Input coordinates (-1 to 1)
        w1: First layer weights (n_hidden x 2 matrix)
        b1: First layer biases (n_hidden element array)
        w2: Second layer weights (n_output x n_hidden matrix) 
        b2: Second layer biases (n_output element array)
        neuron_idx: Which second layer neuron to visualize
        viz_scale: Scaling factor for visualization
    """
    
    # Get number of hidden neurons
    n_hidden = w1.shape[0]
    
    # Compute all first layer outputs
    relu_outputs = []
    for i in range(n_hidden):
        linear_output = w1[i,0] * u + w1[i,1] * v + b1[i]
        relu_output = max(0, linear_output)
        relu_outputs.append(relu_output)
    
    # Second layer neuron computation
    second_layer_output = b2[neuron_idx]
    for i in range(n_hidden):
        second_layer_output += w2[neuron_idx,i] * relu_outputs[i]
    
    # No ReLU on output layer
    # Use output as z-coordinate
    z = second_layer_output * viz_scale
    
    return m.np.array([u, v, z])


def surface_func_second_layer_no_relu(u, v, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.5):
    """
    Surface function for second layer neurons that combines first layer outputs.
    
    Args:
        u, v: Input coordinates (-1 to 1)
        w1: First layer weights (2x2 matrix)
        b1: First layer biases (2 element array)
        w2: Second layer weights (2x2 matrix) 
        b2: Second layer biases (2 element array)
        neuron_idx: Which second layer neuron (0 or 1)
        viz_scale: Scaling factor for visualization
    """
    
    # First layer neuron 1 output
    linear_output_1 = w1[0,0] * u + w1[0,1] * v + b1[0]
    relu_output_1 = max(0, linear_output_1)
    
    # First layer neuron 2 output  
    linear_output_2 = w1[1,0] * u + w1[1,1] * v + b1[1]
    relu_output_2 = max(0, linear_output_2)
    
    # Second layer neuron computation
    second_layer_output = w2[neuron_idx,0] * relu_output_1 + w2[neuron_idx,1] * relu_output_2 + b2[neuron_idx]
    # second_layer_output = max(0, second_layer_output)
    
    # Use output as z-coordinate
    z = second_layer_output * viz_scale
    
    return m.np.array([u, v, z])



def get_second_layer_joints(w1, b1, w2, b2, neuron_idx=0, extent=1):
    """
    Calculate joint lines for second layer neurons.
    
    Returns a list of line segments representing the boundaries where
    the second layer neuron's output changes behavior.
    """
    
    joint_lines = []
    
    # First, get the first layer joint lines (these create discontinuities)
    joint_points_1 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent)
    joint_points_2 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent)
    
    # Add first layer joints as they create surface discontinuities
    if joint_points_1:
        joint_lines.append(joint_points_1)
    if joint_points_2:
        joint_lines.append(joint_points_2)
    
    # Now find where the second layer neuron itself switches on/off
    # This happens when: w2[neuron_idx,0] * relu1 + w2[neuron_idx,1] * relu2 + b2[neuron_idx] = 0
    
    # We need to check this condition in each region defined by the first layer ReLUs
    regions = [
        (False, False),  # Both first layer neurons off
        (True, False),   # Only first neuron on
        (False, True),   # Only second neuron on  
        (True, True)     # Both neurons on
    ]
    
    for relu1_active, relu2_active in regions:
        # Calculate the second layer boundary in this region
        w2_eff = w2[neuron_idx, :]
        b2_eff = b2[neuron_idx]
        
        if relu1_active and relu2_active:
            # Both active: w2[0]*(w1[0,0]*u + w1[0,1]*v + b1[0]) + w2[1]*(w1[1,0]*u + w1[1,1]*v + b1[1]) + b2 = 0
            # Simplifies to: (w2[0]*w1[0,0] + w2[1]*w1[1,0])*u + (w2[0]*w1[0,1] + w2[1]*w1[1,1])*v + (w2[0]*b1[0] + w2[1]*b1[1] + b2) = 0
            eff_w1 = w2_eff[0] * w1[0,0] + w2_eff[1] * w1[1,0]
            eff_w2 = w2_eff[0] * w1[0,1] + w2_eff[1] * w1[1,1]
            eff_b = w2_eff[0] * b1[0] + w2_eff[1] * b1[1] + b2_eff
            
        elif relu1_active and not relu2_active:
            # Only first active: w2[0]*(w1[0,0]*u + w1[0,1]*v + b1[0]) + b2 = 0
            eff_w1 = w2_eff[0] * w1[0,0]
            eff_w2 = w2_eff[0] * w1[0,1]
            eff_b = w2_eff[0] * b1[0] + b2_eff
            
        elif not relu1_active and relu2_active:
            # Only second active: w2[1]*(w1[1,0]*u + w1[1,1]*v + b1[1]) + b2 = 0
            eff_w1 = w2_eff[1] * w1[1,0]
            eff_w2 = w2_eff[1] * w1[1,1]
            eff_b = w2_eff[1] * b1[1] + b2_eff
            
        else:
            # Neither active: b2 = 0
            if abs(b2_eff) < 1e-8:
                # Always on the boundary (degenerate case)
                continue
            else:
                # Never on the boundary in this region
                continue
        
        # Get joint points for this effective linear function
        region_joints = get_relu_joint(eff_w1, eff_w2, eff_b, extent)
        
        if region_joints:
            # Need to clip these joints to the actual region where this applies
            clipped_joints = clip_joint_to_region(region_joints, relu1_active, relu2_active, w1, b1, extent)
            if clipped_joints:
                joint_lines.append(clipped_joints)
    
    return joint_lines

def clip_joint_to_region(joint_points, relu1_active, relu2_active, w1, b1, extent):
    """
    Clip joint line to the region where the specified ReLU conditions hold.
    """
    if not joint_points or len(joint_points) < 2:
        return []
    
    # Sample points along the joint line
    start, end = joint_points[0], joint_points[1]
    clipped_points = []
    
    # Check multiple points along the line
    for t in m.np.linspace(0, 1, 100):
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        
        # Check if this point satisfies the region conditions
        linear1 = w1[0,0] * x + w1[0,1] * y + b1[0]
        linear2 = w1[1,0] * x + w1[1,1] * y + b1[1]
        
        relu1_here = linear1 > 0
        relu2_here = linear2 > 0
        
        if relu1_here == relu1_active and relu2_here == relu2_active:
            clipped_points.append([x, y])
    
    # Return start and end of the clipped segment
    if len(clipped_points) >= 2:
        return [clipped_points[0], clipped_points[-1]]
    else:
        return []

def create_second_layer_joint_lines(w1, b1, w2, b2, neuron_idx=0, extent=1):
    """
    Create Manim objects for all second layer joint lines.
    """
    joint_lines = get_second_layer_joints(w1, b1, w2, b2, neuron_idx, extent)
    
    manim_lines = []
    colors = [m.RED, BLUE, GREEN, YELLOW, m.PURPLE]  # Different colors for different joints
    
    for i, joint_points in enumerate(joint_lines):
        if len(joint_points) >= 2:
            color = colors[i % len(colors)]
            joint_line = m.DashedLine(
                start=[joint_points[0][0], joint_points[0][1], 0],
                end=[joint_points[1][0], joint_points[1][1], 0],
                color=color,
                stroke_width=3,
                dash_length=0.05
            )
            manim_lines.append(joint_line)
    
    return manim_lines


class plane_folding_sketch_1(WelchScene):
    def construct(self):

        #nice trained 2 hidden layer model. 
        w1=m.np.array([[-0.02866297,  1.6250265 ],
             [-1.3056537 ,  0.46831134]], dtype=m.np.float32)
        b1=m.np.array([-0.4677289,  1.0067637], dtype=m.np.float32)
        w2=m.np.array([[ 1.3398709 ,  0.68694556],
                     [-0.29886743, -1.8411286 ]], dtype=m.np.float32)
        b2=m.np.array([-0.7817721 ,  0.90856946], dtype=m.np.float32)
        w3=m.np.array([[ 1.8897862,  3.0432484],
                     [-1.7220999, -2.2057745]], dtype=m.np.float32)
        b3=m.np.array([-1.0249746 ,  0.61326534], dtype=m.np.float32)
        
        #Not using this flat map right now - but will probably want it
        map_img=image_at_height(graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        map_img.scale_to_fit_height(2)  # This sets the height to 2 units (-1 to 1)
        map_img.scale_to_fit_width(2)   # This sets the width to 2 units (-1 to 1)
        map_img.move_to(m.ORIGIN)

        self.set_view(0, 0, 0, (0.03, -0.02, 0.0), 3.27)
        # self.add(map_img)
        # self.wait()

        surface_func_11=partial(surface_func_general, w1=w1[0,0], w2=w1[0,1], b=b1[0], viz_scale=0.3)
        bent_surface_11 = parametric_surface(surface_func_11, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts11=textured_surface(bent_surface_11, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts11, 0,0,0)
        ts11.set_opacity(0.75)
        joint_points_11 = get_relu_joint(w1[0,0], w1[0,1], b1[0], extent=1)
        joint_line_11=line_from_joint_points_1(joint_points_11).set_opacity(0.5)
        group_11=m.Group(ts11, joint_line_11)

        surface_func_12=partial(surface_func_general, w1=w1[1,0], w2=w1[1,1], b=b1[1], viz_scale=0.3)
        bent_surface_12 = parametric_surface(surface_func_12, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts12=textured_surface(bent_surface_12, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts12, 0,0,0)
        ts12.set_opacity(0.75)
        joint_points_12 = get_relu_joint(w1[1,0], w1[1,1], b1[1], extent=1)
        joint_line_12=line_from_joint_points_1(joint_points_12).set_opacity(0.5)
        group_12=m.Group(ts12, joint_line_12)


        # self.add(group_11)
        # self.add(group_12)    
        # group_12.move_to([0, 0, 1.2])
        # self.wait()         

        # Ok scale/flip/add!
        # I almost want liek 2 sets of copies now?
        # One for each combination to come together?
        # Let me start with the first one and then see what's up. 

        neuron_idx=0

        surface_func_21 = partial(
            surface_func_second_layer_no_relu, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=0.3
        )

        # Ok so I'm trying to get my head around the fully polytope of the second layer
        # Claude can't seem to figure it out, and it's close enough to the core thing 
        # i'm trying to understand that i think it's worth me hackign on directly
        # May end up with a better prompts/framing I can pass off, 
        # or maybe I can just figure it out - we'll see!

        bent_surface_21 = parametric_surface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts21 = textured_surface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts21, 0,0,0)
        ts21.set_opacity(0.75)

        self.add(ts21)

        # self.add(joint_line_12, joint_line_11)


        # Ok let me swag for a second here -> the regions before Relu have to be defined by the joints between the 
        # Relu joints in the first layer? Right? Let me try that here. 
        # Ok so I think the move is 
        # Starting to see why claude's solution is pretty long here. 
        # 1. find the intesections between the lines 


        polygons = simple_polygon_finder(joint_points_11, joint_points_12, extent=1)

        polygon_3d_objects = create_3d_polygon_regions(
            polygons, w1, b1, w2, b2, 
            neuron_idx=neuron_idx, viz_scale=0.3
        )

        # Add them to the scene
        for poly in polygon_3d_objects:
            poly.set_opacity(0.3)
            self.add(poly)


        self.wait()


        # Ok, this is great - one interesting thing is that the 2d regions for each neuron are the same, 
        # but the get scaled and clipped differently by ReLu there's proably a cool way to show that
        # Man is there a reality where we show shit moving around during training?!
        # Now before I do the ReLu stuff -> I want to add a 0,0 plane -> the intersections should be exactly
        # where the new polytope borders will be right???


        plane = m.Rectangle(
            width=2,  # -1 to +1 = 2 units wide
            height=2, # -1 to +1 = 2 units tall
            fill_color=m.GREY,
            fill_opacity=0.3,
            stroke_color=m.WHITE,
            stroke_width=1
        )
        plane.move_to([0, 0, 0])  # Position at z=0

        # Add it to your scene
        self.add(plane)
        self.wait()

        #Ok i think i do want to look at pre and post relu together, let's stack
        surface_func_21r = partial(
            surface_func_second_layer, 
            w1=w1, b1=b1, w2=w2, b2=b2, 
            neuron_idx=neuron_idx, viz_scale=0.3
        )

        # Ok so I'm trying to get my head around the fully polytope of the second layer
        # Claude can't seem to figure it out, and it's close enough to the core thing 
        # i'm trying to understand that i think it's worth me hackign on directly
        # May end up with a better prompts/framing I can pass off, 
        # or maybe I can just figure it out - we'll see!

        bent_surface_21r = parametric_surface(surface_func_21r, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        ts21r = textured_surface(bent_surface_21r, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        set_surface_shading(ts21r, 0,0,0)
        ts21r.set_opacity(0.75)
        ts21r.shift([0,0,1.4])

        self.add(ts21r)

        # Ok yeah so naive approach to "just Relu the polygons" doesn't work - why?
        # Right yeah so if my polygon crosses the origin, then it gets a new joint - that's why right?


        polygon_3d_objects_r = create_3d_polygon_regions_with_relu(
            polygons, w1, b1, w2, b2, 
            neuron_idx=neuron_idx, viz_scale=0.3
        )
        # Add them to the scene
        for poly in polygon_3d_objects_r:
            poly.set_opacity(0.3)
            poly.shift([0,0,1.4])
            self.add(poly)

        # self.add(polygon_3d_objects_r[3])

        self.wait()


        # Create colored polygons
        # colors = [RED, BLUE, GREEN, YELLOW, PURPLE]
        # polygon_mobjects = []

        # for i, polygon_points in enumerate(polygons):
        #     if len(polygon_points) >= 3:
        #         # Convert 2D points to 3D for Manim
        #         points_3d = [[p[0], p[1], 0] for p in polygon_points]
                
        #         # Create the polygon
        #         poly = Polygon(*points_3d, 
        #                       fill_color=colors[i % len(colors)], 
        #                       fill_opacity=0.4,
        #                       stroke_color=colors[i % len(colors)],
        #                       stroke_width=2)
                
        #         polygon_mobjects.append(poly)
        #         self.add(poly)

        # # You can also add the joint lines on top
        # # self.add(joint_line_11, joint_line_12)

        # self.wait()

        # bent_surface_21 = ParametricSurface(surface_func_21, u_range=[-1, 1], v_range=[-1, 1], resolution=(50, 50))
        # ts21 = TexturedSurface(bent_surface_21, graphics_dir+'/baarle_hertog_maps/baarle_hertog_maps-11.png')
        # ts21.set_shading(0,0,0)
        # ts21.set_opacity(0.75)

        # joint_lines_21 = create_second_layer_joint_lines(w1, b1, w2, b2, neuron_idx=1, extent=1)
        # group_21_lines = Group(*joint_lines_21)
        # group_21=Group(ts21, group_21_lines)

        # group_21.move_to([0, 0, 2.4])
        # self.add(group_21)  


        # Submerging a surface into a very smooth fully opaque liquid...
        # That's not quite right though, becuase values below the surface get clipped to 0
        # Might be interesting/cool to see shaded regions of the map move around and like come together in 
        # different ways


        # Ok great, bending looks good. And i think oreintation is actually mayby right?
        # Contour lines or heatmaps could be nice -> I think for now just a fold line would be a good starting point

        # self.add(bent_surface)
        
        # Move camera to 3D view
        # self.set_camera_orientation(phi=70*DEGREES, theta=30*DEGREES)
        
        # Animate the transformation from flat to bent
        # self.play(
        #     Transform(map_img, bent_surface),
        #     run_time=3
        # )

        self.wait(20)
        pass











