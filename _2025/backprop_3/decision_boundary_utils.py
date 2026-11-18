import manimgx as m
import numpy as np
from functools import partial
from itertools import combinations
import math
 

def get_decision_boundary_segments(w1, b1, w2, b2, polygons, extent=1):
    """
    Find the decision boundary between two output neurons.
    The boundary occurs where output_0 = output_1.
    
    Args:
        w1, b1: First layer weights and biases
        w2, b2: Second layer weights and biases  
        polygons: List of polygons representing regions
        extent: Domain boundary
        
    Returns:
        List of line segments that form the decision boundary
    """
    
    def evaluate_output_difference(x, y):
        """Evaluate output_0 - output_1 at a point"""
        # Calculate all first layer outputs
        n_hidden = w1.shape[0]
        relu_outputs = []
        
        for i in range(n_hidden):
            linear_output = w1[i,0] * x + w1[i,1] * y + b1[i]
            relu_output = max(0, linear_output)
            relu_outputs.append(relu_output)
        
        # Calculate both outputs
        output_0 = b2[0]
        output_1 = b2[1]
        
        for i in range(n_hidden):
            output_0 += w2[0,i] * relu_outputs[i]
            output_1 += w2[1,i] * relu_outputs[i]
        
        return output_0 - output_1
    
    def find_zero_crossing(p1, p2, tol=1e-6):
        """Find where the line segment p1-p2 crosses the decision boundary"""
        v1 = evaluate_output_difference(p1[0], p1[1])
        v2 = evaluate_output_difference(p2[0], p2[1])
        
        # Check if there's a sign change
        if v1 * v2 > 0:
            return None
            
        # Binary search for the zero crossing
        while np.linalg.norm(np.array(p2) - np.array(p1)) > tol:
            mid = [(p1[0] + p2[0])/2, (p1[1] + p2[1])/2]
            v_mid = evaluate_output_difference(mid[0], mid[1])
            
            if abs(v_mid) < tol:
                return mid
                
            if v1 * v_mid < 0:
                p2 = mid
                v2 = v_mid
            else:
                p1 = mid
                v1 = v_mid
                
        return [(p1[0] + p2[0])/2, (p1[1] + p2[1])/2]
    
    # Find boundary points within each polygon
    boundary_segments = []
    
    for polygon in polygons:
        if len(polygon) < 3:
            continue
            
        polygon_boundary_points = []
        
        # Check each edge of the polygon
        n = len(polygon)
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i+1) % n]
            
            crossing = find_zero_crossing(p1, p2)
            if crossing:
                polygon_boundary_points.append(crossing)
        
        # Also check for boundary points inside the polygon
        # Sample points on a grid within the polygon
        if len(polygon_boundary_points) >= 2:
            # Connect consecutive boundary points
            for i in range(len(polygon_boundary_points) - 1):
                boundary_segments.append([polygon_boundary_points[i], polygon_boundary_points[i+1]])
    
    # Merge segments that connect
    merged_segments = merge_connected_segments(boundary_segments)
    
    return merged_segments


def merge_connected_segments(segments, tol=1e-6):
    """Merge line segments that connect at endpoints"""
    if not segments:
        return []
    
    merged = []
    used = [False] * len(segments)
    
    for i in range(len(segments)):
        if used[i]:
            continue
            
        current_chain = [segments[i][0], segments[i][1]]
        used[i] = True
        
        # Try to extend the chain
        changed = True
        while changed:
            changed = False
            for j in range(len(segments)):
                if used[j]:
                    continue
                    
                # Check if segment j connects to the chain
                seg = segments[j]
                
                # Check all four possible connections
                if np.linalg.norm(np.array(current_chain[-1]) - np.array(seg[0])) < tol:
                    current_chain.append(seg[1])
                    used[j] = True
                    changed = True
                elif np.linalg.norm(np.array(current_chain[-1]) - np.array(seg[1])) < tol:
                    current_chain.append(seg[0])
                    used[j] = True
                    changed = True
                elif np.linalg.norm(np.array(current_chain[0]) - np.array(seg[0])) < tol:
                    current_chain.insert(0, seg[1])
                    used[j] = True
                    changed = True
                elif np.linalg.norm(np.array(current_chain[0]) - np.array(seg[1])) < tol:
                    current_chain.insert(0, seg[0])
                    used[j] = True
                    changed = True
        
        # Add the chain as a single polyline
        if len(current_chain) >= 2:
            merged.append(current_chain)
    
    return merged


def create_decision_boundary_lines(w1, b1, w2, b2, polygons, extent=1, z_offset=0, color=m.WHITE, stroke_width=4):
    """
    Create Manim line objects for the decision boundary.
    
    Args:
        w1, b1: First layer weights and biases
        w2, b2: Second layer weights and biases
        polygons: List of polygons representing regions
        extent: Domain boundary
        z_offset: Height at which to draw the boundary
        color: Color of the boundary lines
        stroke_width: Width of the boundary lines
        
    Returns:
        List of Manim line objects
    """
    boundary_segments = get_decision_boundary_segments(w1, b1, w2, b2, polygons, extent)
    
    line_objects = []
    for segment in boundary_segments:
        if len(segment) >= 2:
            # Convert to 3D points
            points_3d = [[p[0], p[1], z_offset] for p in segment]
            
            if len(points_3d) == 2:
                # Simple line segment
                line = m.Line(points_3d[0], points_3d[1],
                           color=color, 
                           stroke_width=stroke_width)
            else:
                # Polyline
                line = m.VMobject()
                line.set_points_as_corners(points_3d)
                line.set_stroke(color=color, width=stroke_width)
            
            line_objects.append(line)
    
    return line_objects


def get_analytical_decision_boundary(w1, b1, w2, b2, extent=1):
    """
    Get the analytical equation for the decision boundary in each region.
    
    The decision boundary occurs where:
    output_0 - output_1 = 0
    
    This gives us:
    (w2[0,:] - w2[1,:]) · relu_outputs + (b2[0] - b2[1]) = 0
    
    Within each region (defined by which ReLUs are active), this is a linear equation.
    """
    
    # Compute the difference in weights and biases
    w2_diff = w2[0,:] - w2[1,:]  # Shape: (n_hidden,)
    b2_diff = b2[0] - b2[1]
    
    print("Decision boundary analysis:")
    print(f"Weight differences: {w2_diff}")
    print(f"Bias difference: {b2_diff}")
    
    # For each possible activation pattern
    n_hidden = w1.shape[0]
    boundary_equations = []
    
    for pattern in range(2**n_hidden):
        active = []
        for i in range(n_hidden):
            active.append(bool(pattern & (1 << i)))
        
        # In this region, the decision boundary is:
        # sum over active neurons of: w2_diff[i] * (w1[i,0]*x + w1[i,1]*y + b1[i]) + b2_diff = 0
        
        a = 0  # coefficient of x
        b = 0  # coefficient of y  
        c = b2_diff  # constant term
        
        for i in range(n_hidden):
            if active[i]:
                a += w2_diff[i] * w1[i,0]
                b += w2_diff[i] * w1[i,1]
                c += w2_diff[i] * b1[i]
        
        if abs(a) > 1e-10 or abs(b) > 1e-10:
            # Non-degenerate line
            boundary_equations.append({
                'pattern': active,
                'equation': (a, b, c),  # ax + by + c = 0
                'description': f"{a:.3f}x + {b:.3f}y + {c:.3f} = 0"
            })
    
    return boundary_equations