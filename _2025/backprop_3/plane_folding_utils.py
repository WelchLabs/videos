import manimgx as m
import numpy as np
from functools import partial
from itertools import combinations
import math
 
CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'


def get_polygon_corners_multi(joint_points_list, extent=1):
    """
    Compute the corner points of polygons formed by multiple ReLU joint lines.
    Each line divides the plane into two half-planes, creating distinct regions.
    
    Args:
        joint_points_list: List of joint point pairs, one for each neuron
        extent: The boundary of the square domain
    
    Returns:
        List of polygons, where each polygon is a list of corner points
    """
    # Filter out empty joint lines
    valid_lines = []
    for joint_points in joint_points_list:
        if joint_points and len(joint_points) >= 2:
            valid_lines.append(joint_points[:2])
    
    if len(valid_lines) == 0:
        # If no lines, return the entire square
        return [[[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]]]
    
    def line_equation(p1, p2):
        """Get line equation coefficients ax + by + c = 0"""
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        return a, b, c
    
    def evaluate_line(point, a, b, c):
        """Evaluate ax + by + c for a point"""
        return a * point[0] + b * point[1] + c
    
    def line_intersection(p1, p2, p3, p4):
        """Find intersection of two lines"""
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
    
    def clip_polygon_by_line(polygon, line_p1, line_p2):
        """Clip a polygon by a line using Sutherland-Hodgman algorithm"""
        if not polygon:
            return []
        
        a, b, c = line_equation(line_p1, line_p2)
        
        output_polygon = []
        n = len(polygon)
        
        for i in range(n):
            current_vertex = polygon[i]
            previous_vertex = polygon[i-1]
            
            current_side = evaluate_line(current_vertex, a, b, c)
            previous_side = evaluate_line(previous_vertex, a, b, c)
            
            # We keep points on the positive side (or on the line)
            if current_side >= -1e-10:  # Current vertex is inside
                if previous_side < -1e-10:  # Previous vertex was outside
                    # Add intersection point
                    intersection = line_intersection(
                        previous_vertex, current_vertex,
                        line_p1, line_p2
                    )
                    if intersection:
                        output_polygon.append(intersection)
                output_polygon.append(current_vertex)
            elif previous_side >= -1e-10:  # Current is outside, previous was inside
                # Add intersection point
                intersection = line_intersection(
                    previous_vertex, current_vertex,
                    line_p1, line_p2
                )
                if intersection:
                    output_polygon.append(intersection)
        
        return output_polygon
    
    # Start with the full square
    initial_square = [
        [-extent, -extent],
        [extent, -extent],
        [extent, extent],
        [-extent, extent]
    ]
    
    # For each combination of line sides, create a region
    n_lines = len(valid_lines)
    polygons = []
    
    for region_idx in range(2**n_lines):
        # Start with the full square
        current_polygon = initial_square[:]
        
        # Clip by each line
        for line_idx in range(n_lines):
            if current_polygon:
                line = valid_lines[line_idx]
                
                # Determine which side to keep based on the region code
                if region_idx & (1 << line_idx):
                    # Keep positive side - use line as is
                    current_polygon = clip_polygon_by_line(current_polygon, line[0], line[1])
                else:
                    # Keep negative side - reverse line direction
                    current_polygon = clip_polygon_by_line(current_polygon, line[1], line[0])
        
        # Add the resulting polygon if it's non-empty
        if len(current_polygon) >= 3:
            # Remove duplicate vertices
            cleaned_polygon = []
            for vertex in current_polygon:
                is_duplicate = False
                for existing in cleaned_polygon:
                    if abs(vertex[0] - existing[0]) < 1e-8 and abs(vertex[1] - existing[1]) < 1e-8:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    cleaned_polygon.append(vertex)
            
            if len(cleaned_polygon) >= 3:
                polygons.append(cleaned_polygon)
    
    return polygons


def create_3d_polygon_regions_multi_wth_relu(polygons, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.3):
    """
    Create 3D polygons by mapping each corner point to its second layer output height.
    Works with any number of hidden neurons.
    
    Args:
        polygons: List of 2D polygon corner points from get_polygon_corners_multi
        w1, b1: First layer weights and biases (shape: [n_hidden, 2] and [n_hidden])
        w2, b2: Second layer weights and biases (shape: [n_output, n_hidden] and [n_output])
        neuron_idx: Which second layer neuron to visualize
        viz_scale: Scaling factor for z-coordinate
    
    Returns:
        List of 3D polygon objects
    """
    
    def evaluate_second_layer_at_point(x, y):
        """Evaluate the second layer neuron output at a specific (x,y) point"""
        # Calculate all first layer outputs
        n_hidden = w1.shape[0]
        relu_outputs = []
        
        for i in range(n_hidden):
            linear_output = w1[i,0] * x + w1[i,1] * y + b1[i]
            relu_output = max(0, linear_output)
            relu_outputs.append(relu_output)
        
        # Second layer output (no ReLU applied here to see full surface)
        second_layer_output = b2[neuron_idx]
        for i in range(n_hidden):
            second_layer_output += w2[neuron_idx,i] * relu_outputs[i]

        second_layer_output = max(0, second_layer_output)

        return second_layer_output * viz_scale
    
    polygon_objects = []
    colors = [m.RED, BLUE, GREEN, YELLOW, m.PURPLE, m.ORANGE, m.PINK, m.TEAL]
    
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


def create_3d_polygon_regions_multi(polygons, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.3):
    """
    Create 3D polygons by mapping each corner point to its second layer output height.
    Works with any number of hidden neurons.
    
    Args:
        polygons: List of 2D polygon corner points from get_polygon_corners_multi
        w1, b1: First layer weights and biases (shape: [n_hidden, 2] and [n_hidden])
        w2, b2: Second layer weights and biases (shape: [n_output, n_hidden] and [n_output])
        neuron_idx: Which second layer neuron to visualize
        viz_scale: Scaling factor for z-coordinate
    
    Returns:
        List of 3D polygon objects
    """
    
    def evaluate_second_layer_at_point(x, y):
        """Evaluate the second layer neuron output at a specific (x,y) point"""
        # Calculate all first layer outputs
        n_hidden = w1.shape[0]
        relu_outputs = []
        
        for i in range(n_hidden):
            linear_output = w1[i,0] * x + w1[i,1] * y + b1[i]
            relu_output = max(0, linear_output)
            relu_outputs.append(relu_output)
        
        # Second layer output (no ReLU applied here to see full surface)
        second_layer_output = b2[neuron_idx]
        for i in range(n_hidden):
            second_layer_output += w2[neuron_idx,i] * relu_outputs[i]
        
        return second_layer_output * viz_scale
    
    polygon_objects = []
    colors = [m.RED, BLUE, GREEN, YELLOW, m.PURPLE, m.ORANGE, m.PINK, m.TEAL]
    
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
    Compute the corner points of polygons formed by two ReLU joint lines.
    
    This version uses a direct geometric approach to construct exactly 4 regions.
    """
    if not joint_points_1 or not joint_points_2:
        return []
    
    if len(joint_points_1) < 2 or len(joint_points_2) < 2:
        return []
    
    line1 = joint_points_1[:2]
    line2 = joint_points_2[:2] 
    
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
    
    def get_line_boundary_intersections(line, extent):
        """Get intersections of line with square boundary, properly sorted"""
        p1, p2 = line
        intersections = []
        
        # Check intersection with each boundary edge
        boundaries = [
            [[-extent, -extent], [extent, -extent]],  # bottom
            [[extent, -extent], [extent, extent]],    # right  
            [[extent, extent], [-extent, extent]],    # top
            [[-extent, extent], [-extent, -extent]]   # left
        ]
        
        for boundary in boundaries:
            intersection = line_intersection(p1, p2, boundary[0], boundary[1])
            if intersection is not None:
                x, y = intersection
                # Check if intersection is within boundary segment
                if (-extent-1e-8 <= x <= extent+1e-8 and -extent-1e-8 <= y <= extent+1e-8):
                    # Snap to exact boundary values
                    if abs(x - extent) < 1e-6: x = extent
                    elif abs(x + extent) < 1e-6: x = -extent
                    if abs(y - extent) < 1e-6: y = extent
                    elif abs(y + extent) < 1e-6: y = -extent
                    intersections.append([x, y])
        
        # Remove duplicates and sort along the line
        unique_intersections = []
        for pt in intersections:
            is_duplicate = False
            for existing in unique_intersections:
                if abs(pt[0] - existing[0]) < 1e-8 and abs(pt[1] - existing[1]) < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(pt)
        
        return unique_intersections
    
    # Find intersection of the two lines
    intersection = line_intersection(line1[0], line1[1], line2[0], line2[1])
    if intersection is None:
        return []
    
    # Get boundary intersections
    line1_boundary = get_line_boundary_intersections(line1, extent)
    line2_boundary = get_line_boundary_intersections(line2, extent)
    
    if len(line1_boundary) != 2 or len(line2_boundary) != 2:
        return []  # Each line should intersect boundary at exactly 2 points
    
    # Label the boundary points
    line1_pt1, line1_pt2 = line1_boundary[0], line1_boundary[1]
    line2_pt1, line2_pt2 = line2_boundary[0], line2_boundary[1]
    
    # Define the four corner points of the square
    corners = {
        'bl': [-extent, -extent],  # bottom-left
        'br': [extent, -extent],   # bottom-right  
        'tr': [extent, extent],    # top-right
        'tl': [-extent, extent]    # top-left
    }
    
    def point_on_boundary_between(start_corner, end_corner, boundary_points):
        """Find boundary points that lie between two corners"""
        # This is a simplified approach - we'll determine this based on coordinates
        result = []
        
        start_x, start_y = corners[start_corner]
        end_x, end_y = corners[end_corner]
        
        for pt in boundary_points:
            x, y = pt
            
            # Check if point lies on the boundary segment between start and end corners
            if start_corner == 'bl' and end_corner == 'br':  # bottom edge
                if abs(y + extent) < 1e-6 and start_x <= x <= end_x:
                    result.append(pt)
            elif start_corner == 'br' and end_corner == 'tr':  # right edge  
                if abs(x - extent) < 1e-6 and start_y <= y <= end_y:
                    result.append(pt)
            elif start_corner == 'tr' and end_corner == 'tl':  # top edge
                if abs(y - extent) < 1e-6 and end_x <= x <= start_x:
                    result.append(pt)
            elif start_corner == 'tl' and end_corner == 'bl':  # left edge
                if abs(x + extent) < 1e-6 and end_y <= y <= start_y:
                    result.append(pt)
        
        return result
    
    # Now construct each region by walking around its boundary
    # We'll construct them systematically by determining which corners and boundary points belong to each
    
    def side_of_line(point, line_start, line_end):
        """Determine which side of a line a point is on"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    
    # Classify each corner
    corner_classifications = {}
    for corner_name, corner_pos in corners.items():
        side1 = side_of_line(corner_pos, line1[0], line1[1])
        side2 = side_of_line(corner_pos, line2[0], line2[1])
        region = ('+' if side1 > 0 else '-') + ('+' if side2 > 0 else '-')
        corner_classifications[corner_name] = region
    
    # Build regions
    regions = {
        '++': [],
        '+-': [], 
        '--': [],
        '-+': []
    }
    
    # Add corners to their regions
    for corner_name, region in corner_classifications.items():
        regions[region].append(corners[corner_name])
    
    # Add intersection point to all regions
    for region in regions.values():
        region.append(intersection)
    
    # Add boundary intersections to appropriate regions
    all_boundary_points = line1_boundary + line2_boundary
    
    for boundary_pt in all_boundary_points:
        side1 = side_of_line(boundary_pt, line1[0], line1[1])
        side2 = side_of_line(boundary_pt, line2[0], line2[1])
        
        # If point is on line1, add to regions on both sides of line1
        if abs(side1) < 1e-8:
            if side2 >= 0:
                regions['++'].append(boundary_pt)
                regions['-+'].append(boundary_pt)
            else:
                regions['+-'].append(boundary_pt)
                regions['--'].append(boundary_pt)
        # If point is on line2, add to regions on both sides of line2  
        elif abs(side2) < 1e-8:
            if side1 >= 0:
                regions['++'].append(boundary_pt)
                regions['+-'].append(boundary_pt)
            else:
                regions['-+'].append(boundary_pt)
                regions['--'].append(boundary_pt)
    
    # Clean up and order each region
    polygons = []
    
    for region_key, points in regions.items():
        if len(points) < 3:
            continue
            
        # Remove duplicates
        unique_points = []
        for point in points:
            is_duplicate = False
            for existing in unique_points:
                if (abs(point[0] - existing[0]) < 1e-8 and 
                    abs(point[1] - existing[1]) < 1e-8):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        if len(unique_points) >= 3:
            # Calculate centroid
            cx = sum(p[0] for p in unique_points) / len(unique_points)
            cy = sum(p[1] for p in unique_points) / len(unique_points)
            
            # Sort by angle around centroid
            import math
            def angle_from_centroid(point):
                return math.atan2(point[1] - cy, point[0] - cx)
            
            ordered_points = sorted(unique_points, key=angle_from_centroid)
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
    if np.abs(weight_2) < 1e-8: 
        x_intercept = -bias / weight_1
        return [[x_intercept, -extent], [x_intercept, extent]] if -extent <= x_intercept <= extent else []
    elif np.abs(weight_1) < 1e-8:
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

def line_from_joint_points_1(joint_points, stroke_width=3, color=m.WHITE):
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
                color=color,
                stroke_width=stroke_width,
                dash_length=0.08
            )
            return joint_line


def surface_func_general(u, v, w1, w2, b, viz_scale=0.5):
    linear_output = w1 * u + w2 * v + b
    relu_output = max(0, linear_output)
    z = relu_output * viz_scale 
    return np.array([u, v, z])


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
    
    return np.array([u, v, z])


def surface_func_second_layer_multi(u, v, w1, b1, w2, b2, neuron_idx=0, viz_scale=0.5):
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
    
    second_layer_output = max(0, second_layer_output) #Relu Bra!

    # Use output as z-coordinate
    z = second_layer_output * viz_scale
    
    return np.array([u, v, z])


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
    
    return np.array([u, v, z])


def surface_func_third_layer_no_relu_multi(u, v, w1, b1, w2, b2, w3, b3, neuron_idx=0, viz_scale=0.5):
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
    # n_hidden = w1.shape[0]
    
    # Compute all first layer outputs
    relu_outputs = []
    for i in range(w1.shape[0]):
        linear_output = w1[i,0] * u + w1[i,1] * v + b1[i]
        relu_output = max(0, linear_output)
        relu_outputs.append(relu_output)
    
    # Second layer neuron computation
    # second_layer_outputs=
    # Hmm ok there's got to be a better way to do this -> this is super clunky/manual
    # I know i'm on a tight schedule, but I think a little refactor woudl proably have some pretty big
    # beenfits.
    second_layer_output = b2[neuron_idx]
    for i in range(w2.shape[0]):
        second_layer_output += w2[neuron_idx,i] * relu_outputs[i]

    second_layer_output = max(0, second_layer_output)
    
    third_layer_output = b3[neuron_idx]
    for i in range(w3.shape[0]):
        third_layer_output += w3[neuron_idx,i] * second_layer_output[i]

    # No ReLU on output layer
    # Use output as z-coordinate
    z = third_layer_output * viz_scale
    
    return np.array([u, v, z])



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
    
    return np.array([u, v, z])



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
    for t in np.linspace(0, 1, 100):
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