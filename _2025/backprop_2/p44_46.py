from welch_paths import welch_path
import manimgx as m
from welch_geometry import diagram_axes
from welch_surface import SampledSurface
from welch_scene import WelchScene, image_at_height
from welch_text import math_tex
from functools import partial
import numpy as np
import torch

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

svg_path=welch_path('backprop2/graphics/to_manim')
data_path=welch_path('backprop2/hackin')
heatmap_path=welch_path('backprop2/graphics/to_manim/jun_6_1')

# map_min_x=0.38
# map_max_x=1.54
# map_min_y=-0.56
# map_max_y=0.56

# min_long=-7.0
# max_long=18.0
# min_lat=36.0
# max_lat=56.0


def format_number(num, total_chars=6, align='right'):
    """
    Format number to maintain consistent visual alignment for animations.
    
    Args:
        num: The number to format
        total_chars: Total character width (should accommodate largest expected number)
        align: 'right', 'left', or 'center' - how to align within the fixed width
    """
    abs_num = abs(num)
    
    # Determine appropriate precision based on magnitude
    if abs_num >= 100:
        # 100+: no decimal places (e.g., "123", "-123")
        formatted = f"{num:.0f}"
    elif abs_num >= 10:
        # 10-99: one decimal place (e.g., "12.3", "-12.3")  
        formatted = f"{num:.1f}"
    elif abs_num >= 1:
        # 1-9: two decimal places (e.g., "1.23", "-1.23")
        formatted = f"{num:.2f}"
    else:
        # Less than 1: two decimal places (e.g., "0.12", "-0.12")
        formatted = f"{num:.2f}"
    
    # Pad to consistent width
    if align == 'right':
        return formatted.rjust(total_chars)
    elif align == 'left':
        return formatted.ljust(total_chars)
    else:  # center
        return formatted.center(total_chars)

def format_number_fixed_decimal(num, decimal_places=2, total_chars=6):
    """
    Alternative formatter that keeps decimal point in same position.
    Useful when you want all numbers to have the same decimal precision.
    """
    formatted = f"{num:.{decimal_places}f}"
    return formatted.rjust(total_chars)

def get_numbers(i, xs, weights, logits, yhats):
    x = xs[i, -1]
    numbers = m.VGroup()

    tx = math_tex(str(x) + r'^\circ')
    tx.scale(0.13)
    tx.move_to([-1.49, 0.02, 0])
    numbers.add(tx)
    
    # Weights - using consistent formatting
    w = weights[i, :]
    tm1 = math_tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
    tm1.scale(0.16)
    tm1.move_to([-1.195, 0.205, 0])
    numbers.add(tm1)
    
    tm2 = math_tex(format_number(w[1], total_chars=6)).set_color(YELLOW)
    tm2.scale(0.15)
    tm2.move_to([-1.155, 0.015, 0])
    numbers.add(tm2)
    
    tm3 = math_tex(format_number(w[2], total_chars=6)).set_color(GREEN)
    tm3.scale(0.16)
    tm3.move_to([-1.19, -0.17, 0])
    numbers.add(tm3)
    
    # Biases
    tb1 = math_tex(format_number(w[3], total_chars=6)).set_color('#00FFFF')
    tb1.scale(0.16)
    tb1.move_to([-0.875, 0.365, 0])
    numbers.add(tb1)
    
    tb2 = math_tex(format_number(w[4], total_chars=6)).set_color(YELLOW)
    tb2.scale(0.16)
    tb2.move_to([-0.875, 0.015, 0])
    numbers.add(tb2)
    
    tb3 = math_tex(format_number(w[5], total_chars=6)).set_color(GREEN)
    tb3.scale(0.16)
    tb3.move_to([-0.88, -0.335, 0])
    numbers.add(tb3)
    
    # Logits
    tl1 = math_tex(format_number(logits[i, 0], total_chars=6)).set_color('#00FFFF')
    tl1.scale(0.16)
    tl1.move_to([-0.52, 0.37, 0])
    numbers.add(tl1)
    
    tl2 = math_tex(format_number(logits[i, 1], total_chars=6)).set_color(YELLOW)
    tl2.scale(0.16)
    tl2.move_to([-0.52, 0.015, 0])
    numbers.add(tl2)
    
    tl3 = math_tex(format_number(logits[i, 2], total_chars=6)).set_color(GREEN)
    tl3.scale(0.16)  
    tl3.move_to([-0.52, -0.335, 0])
    numbers.add(tl3)
    
    # Predictions
    yhat1 = math_tex(format_number(yhats[i, 0], total_chars=6)).set_color('#00FFFF')
    yhat1.scale(0.16)
    yhat1.move_to([0.22, 0.37, 0])
    numbers.add(yhat1)
    
    yhat2 = math_tex(format_number(yhats[i, 1], total_chars=6)).set_color(YELLOW)
    yhat2.scale(0.16)
    yhat2.move_to([0.22, 0.015, 0])
    numbers.add(yhat2)
    
    yhat3 = math_tex(format_number(yhats[i, 2], total_chars=6)).set_color(GREEN)
    yhat3.scale(0.16)
    yhat3.move_to([0.22, -0.335, 0])
    numbers.add(yhat3)
    
    return numbers



def latlong_to_canvas(lat, long, 
                      label=None,
                      map_min_x=0.38, map_max_x=1.54,
                      map_min_y=-0.56, map_max_y=0.56,
                      min_long=-7.0, max_long=18.0,
                      min_lat=36.0, max_lat=56.0,
                      paris_adjust=[0,0,0],
                      madrid_adjust=[0,0,0],
                      berlin_adjust=[-0.03, 0.06, 0], 
                      barcelona_adjust=[0,0,0]):
    """
    Convert latitude/longitude coordinates to canvas x,y coordinates.
    
    Args:
        lat: Latitude value
        long: Longitude value
        map_min_x, map_max_x: Canvas x-coordinate bounds
        map_min_y, map_max_y: Canvas y-coordinate bounds
        min_long, max_long: Longitude bounds for the map
        min_lat, max_lat: Latitude bounds for the map
    
    Returns:
        tuple: (x, y) canvas coordinates
    """
    # Normalize longitude to [0, 1] range
    long_normalized = (long - min_long) / (max_long - min_long)
    
    # Normalize latitude to [0, 1] range
    lat_normalized = (lat - min_lat) / (max_lat - min_lat)
    
    # Map to canvas coordinates
    x = map_min_x + long_normalized * (map_max_x - map_min_x)
    y = map_min_y + lat_normalized * (map_max_y - map_min_y)

    if label is not None:
        if label==0: 
            x=x+madrid_adjust[0]; y=y+madrid_adjust[1]
        if label==1: 
            x=x+paris_adjust[0]; y=y+paris_adjust[1]
        if label==2: 
            x=x+berlin_adjust[0]; y=y+berlin_adjust[1]
        if label==3: 
            x=x+barcelona_adjust[0]; y=y+barcelona_adjust[1]

    return x, y


def get_grad_regions(i, ys, yhats, grads):
     #Ok let's shade some lines!
    max_region_width=0.15
    min_region_width=0.01
    region_scaling=0.15

    grad_regions = m.VGroup()

    y_one_hot=torch.nn.functional.one_hot(torch.tensor(int(ys[i])),3).numpy()
    dldh=yhats[i]-y_one_hot

    rh1_width=np.clip(region_scaling*np.abs(dldh[0]), min_region_width, max_region_width)
    rh1=m.Rectangle(0.425, rh1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
    rh1.move_to([-0.52, 0.37, 0])
    grad_regions.add(rh1)

    rh2_width=np.clip(region_scaling*np.abs(dldh[1]), min_region_width, max_region_width)
    rh2=m.Rectangle(0.425, rh2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
    rh2.move_to([-0.52, 0.015, 0])
    grad_regions.add(rh2)

    rh3_width=np.clip(region_scaling*np.abs(dldh[2]), min_region_width, max_region_width)
    rh3=m.Rectangle(0.425, rh3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
    rh3.move_to([-0.52, -0.335, 0])
    grad_regions.add(rh3)

    rb1_width=np.clip(region_scaling*np.abs(grads[i,3]), min_region_width, max_region_width)
    rb1=m.Rectangle(0.24, rb1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
    rb1.move_to([-0.875, 0.37, 0])
    grad_regions.add(rb1)

    rb2_width=np.clip(region_scaling*np.abs(grads[i,4]), min_region_width, max_region_width)
    rb2=m.Rectangle(0.24, rb2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
    rb2.move_to([-0.875, 0.015, 0])
    grad_regions.add(rb2)

    rb3_width=np.clip(region_scaling*np.abs(grads[i,5]), min_region_width, max_region_width)
    rb3=m.Rectangle(0.24, rb3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
    rb3.move_to([-0.872, -0.335, 0])
    grad_regions.add(rb3)

    rm1_width=np.clip(region_scaling*np.abs(grads[i,0]), min_region_width, max_region_width)
    rm1=m.Rectangle(0.42, rm1_width, stroke_width=0).set_color('#00FFFF').set_opacity(0.2)
    rm1.rotate(33*m.DEGREES)
    rm1.move_to([-1.18, 0.20, 0])
    grad_regions.add(rm1)

    rm2_width=np.clip(region_scaling*np.abs(grads[i,1]), min_region_width, max_region_width)
    rm2=m.Rectangle(0.33, rm2_width, stroke_width=0).set_color(YELLOW).set_opacity(0.2)
    # rm2.rotate(33*DEGREES)
    rm2.move_to([-1.18, 0.015, 0])
    grad_regions.add(rm2)

    rm3_width=np.clip(region_scaling*np.abs(grads[i,2]), min_region_width, max_region_width)
    rm3=m.Rectangle(0.42, rm3_width, stroke_width=0).set_color(GREEN).set_opacity(0.2)
    rm3.rotate(-30.5*m.DEGREES)
    rm3.move_to([-1.19, -0.175, 0])
    grad_regions.add(rm3)

    return grad_regions

def get_arrow_tip(line, color=None, scale=0.1, tip_position=1.0):
    """
    Add an arrow tip to the end of a line/curve.
    
    Args:
        line: The line/curve object to add arrow to
        color: Color of the arrow tip (defaults to line's color)
        scale: Size of the arrow tip (default 0.1)
        tip_position: Position along the line (0.0 to 1.0, default 1.0 for end)
    
    Returns:
        The ArrowTip object
    """
    # Get the tip point and direction
    tip_point = line.point_from_proportion(tip_position)
    direction_point = line.point_from_proportion(tip_position - 0.05)
    direction = tip_point - direction_point
    
    # Create and position arrow tip
    arrow_tip = m.ArrowTip().scale(scale)
    if color is None:
        color = line.get_color()
    arrow_tip.set_color(color)
    arrow_tip.move_to(tip_point)
    arrow_tip.rotate(m.angle_of_vector(direction))
    
    return arrow_tip

# Create planes from line endpoints, extending only in positive y direction

def create_plane_from_line_endpoints(line, color, depth=3.0, y_extension=2.0):
    """
    Create a plane from the endpoints of an existing line object.
    The plane extends in positive y direction and in z direction (depth).
    
    Args:
        line: The existing line object
        color: Color for the plane
        depth: How far to extend in z direction (both + and -)
        y_extension: How far to extend in positive y direction
    """
    # Get the endpoints of the line
    start_point = line.get_start()
    end_point = line.get_end()
    
    # Create the four corners of our plane
    # Bottom edge (original line endpoints)
    bottom_left = start_point.copy()
    bottom_right = end_point.copy()
    
    # Top edge (extended in positive y direction)
    top_left = start_point + np.array([0, y_extension, 0])
    top_right = end_point + np.array([0, y_extension, 0])
    
    # Create a custom surface class for this rectangular plane
    class RectangularPlane(SampledSurface):
        def __init__(self, corners, **kwargs):
            self.corners = corners
            super().__init__(
                u_range=(0, 1),
                v_range=(0, 1), 
                resolution=(20, 10),
                **kwargs
            )
        
        def uv_func(self, u, v):
            # u goes from left to right (along the line)
            # v goes from bottom to top (y extension)
            
            # Interpolate along bottom edge
            bottom_point = m.interpolate(self.corners[0], self.corners[1], u)
            # Interpolate along top edge  
            top_point = m.interpolate(self.corners[2], self.corners[3], u)
            # Interpolate between bottom and top
            point = m.interpolate(bottom_point, top_point, v)
            
            return point
    
    # Create the base plane (no depth yet)
    base_plane = RectangularPlane(
        [bottom_left, bottom_right, top_left, top_right],
        color=color,
        shading=(0.2, 0.2, 0.6)
    )
    
    # Now extrude this plane in the z direction to give it depth
    class ExtrudedPlane(SampledSurface):
        def __init__(self, base_corners, depth, **kwargs):
            self.base_corners = base_corners
            self.depth = depth
            super().__init__(
                u_range=(0, 1),
                v_range=(-1, 1),  # -1 to 1 to go from -depth/2 to +depth/2
                resolution=(20, 10),
                **kwargs
            )
        
        def uv_func(self, u, v):
            # u parameter: interpolate along the line (left to right)
            # v parameter: interpolate in depth (z direction)
            
            # Get bottom and top points along the line at parameter u
            bottom_point = m.interpolate(self.base_corners[0], self.base_corners[1], u)
            top_point = m.interpolate(self.base_corners[2], self.base_corners[3], u)
            
            # Interpolate between bottom and top based on y-extension
            # For now, let's use the middle height
            middle_point = m.interpolate(bottom_point, top_point, 0.5)
            
            # Add depth in z direction
            z_offset = v * self.depth / 2
            final_point = middle_point + np.array([0, 0, z_offset])
            
            return final_point
    
    # Actually, let's make this simpler - create a plane that extends from the line
    class LineExtensionPlane(SampledSurface):
        def __init__(self, line_start, line_end, y_extension, depth, **kwargs):
            self.line_start = line_start
            self.line_end = line_end  
            self.y_extension = y_extension
            self.depth = depth
            super().__init__(
                u_range=(0, 1),  # Along the line
                v_range=(0, 1),  # From line to extended height
                resolution=(15, 10),
                **kwargs
            )
        
        def uv_func(self, u, v):
            # u: interpolate along original line
            line_point = m.interpolate(self.line_start, self.line_end, u)
            
            # v: extend in positive y direction
            extended_point = line_point + np.array([0, v * self.y_extension, 0])
            
            return extended_point
    
    return LineExtensionPlane(
        start_point, end_point, y_extension, depth,
        color=color,
        shading=(0.2, 0.2, 0.6)
    )


def sample_points_from_curve(curve, num_points=128):
    """
    Sample points from an existing manim curve object
    
    Args:
        curve: The manim curve object (like your softmax_curve_1)
        num_points: Number of points to sample along the curve
    
    Returns:
        Array of 3D points sampled from the curve
    """
    points = []
    for i in range(num_points):
        # Get parameter from 0 to 1 along the curve
        t = i / (num_points - 1)
        # Get the 3D point at this parameter
        point = curve.point_from_proportion(t)
        points.append(point)
    
    return np.array(points)

def create_surface_from_curve_simple(curve, y_extension=0.5, color='#00FFFF'):
    """
    Simpler approach - sample curve points and create surface directly
    """
    # Get curve points
    curve_points = sample_points_from_curve(curve, num_points=32)
    
    # Create extended points (top edge of surface)
    extended_points = curve_points + np.array([0, y_extension, 0])
    
    class SimpleCurveSurface(SampledSurface):
        def __init__(self, bottom_points, top_points, **kwargs):
            self.bottom_points = bottom_points
            self.top_points = top_points
            self.num_points = len(bottom_points)
            
            super().__init__(
                u_range=(0, 1),  # Along curve
                v_range=(0, 1),  # From bottom to top
                resolution=(self.num_points, 8),
                **kwargs
            )
        
        def uv_func(self, u, v):
            # Find position along curve using u
            point_index = u * (self.num_points - 1)
            index_low = int(np.floor(point_index))
            index_high = min(index_low + 1, self.num_points - 1)
            t = point_index - index_low
            
            # Interpolate bottom edge point
            if index_low == index_high:
                bottom_point = self.bottom_points[index_low]
                top_point = self.top_points[index_low]
            else:
                bottom_point = (1-t) * self.bottom_points[index_low] + t * self.bottom_points[index_high]
                top_point = (1-t) * self.top_points[index_low] + t * self.top_points[index_high]
            
            # Interpolate between bottom and top using v
            final_point = (1-v) * bottom_point + v * top_point
            
            return final_point
    
    surface = SimpleCurveSurface(
        bottom_points=curve_points,
        top_points=extended_points,
        color=color,
        shading=(0.2, 0.2, 0.6)
    )
    surface.set_opacity(0.3)
    return surface

def create_matching_plane_and_surface(line, curve, y_extension=0.5, color='#00FFFF'):
    """
    Create plane and surface with matching resolutions for smooth transformation
    """
    # Sample points from curve for consistent structure
    curve_points = sample_points_from_curve(curve, num_points=32)
    
    # Create plane with same resolution as surface
    class MatchingPlane(SampledSurface):
        def __init__(self, line_start, line_end, y_extension, **kwargs):
            self.line_start = line_start
            self.line_end = line_end
            self.y_extension = y_extension
            
            super().__init__(
                u_range=(0, 1),
                v_range=(0, 1),
                resolution=(32, 8),  # Match surface resolution
                **kwargs
            )
        
        def uv_func(self, u, v):
            # u: along the line
            line_point = (1-u) * self.line_start + u * self.line_end
            # v: extend upward
            extended_point = line_point + np.array([0, v * self.y_extension, 0])
            return extended_point
    
    # Create surface with same resolution
    class MatchingSurface(SampledSurface):
        def __init__(self, curve_points, y_extension, **kwargs):
            self.curve_points = curve_points
            self.y_extension = y_extension
            self.num_points = len(curve_points)
            
            super().__init__(
                u_range=(0, 1),
                v_range=(0, 1),
                resolution=(32, 8),  # Same as plane
                **kwargs
            )
        
        def uv_func(self, u, v):
            # Interpolate along curve points
            point_index = u * (self.num_points - 1)
            index_low = int(np.floor(point_index))
            index_high = min(index_low + 1, self.num_points - 1)
            t = point_index - index_low
            
            if index_low == index_high:
                curve_point = self.curve_points[index_low]
            else:
                curve_point = (1-t) * self.curve_points[index_low] + t * self.curve_points[index_high]
            
            # Extend upward
            extended_point = curve_point + np.array([0, v * self.y_extension, 0])
            return extended_point
    
    # Create both objects
    plane = MatchingPlane(
        line.get_start(), line.get_end(), y_extension,
        color=color, shading=(0.2, 0.2, 0.6)
    )
    
    surface = MatchingSurface(
        curve_points, y_extension,
        color=color, shading=(0.2, 0.2, 0.6)
    )
    
    plane.set_opacity(0.3)
    surface.set_opacity(0.3)
    
    return plane, surface


class p44(WelchScene):
    def construct(self):

        min_long=-9.8
        max_long=17.2
        min_lat=36.15 
        max_lat=54.7 

        data=np.load(data_path+'/cities_1d_5.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]


        net_background=m.SVGMobject(svg_path+'/p44_background_1.svg')
        
        i=0
        nums=get_numbers(i, xs, weights, logits, yhats)
        grad_regions=get_grad_regions(i, ys, yhats, grads)
        heatmaps=m.Group()
        heatmap_yhat3=image_at_height(heatmap_path +'/'+str(i)+'_yhat_3.png')
        heatmap_yhat3.scale([0.29, 0.28, 0.28])
        heatmap_yhat3.move_to([0.96,0,0])
        heatmap_yhat3.set_opacity(0.5)
        heatmaps.add(heatmap_yhat3)

        heatmap_yhat1=image_at_height(heatmap_path +'/'+str(i)+'_yhat_1.png')
        heatmap_yhat1.scale([0.29, 0.28, 0.28])
        heatmap_yhat1.move_to([0.96,0,0])
        heatmap_yhat1.set_opacity(0.5)
        heatmaps.add(heatmap_yhat1)

        heatmap_yhat2=image_at_height(heatmap_path +'/'+str(i)+'_yhat_2.png')
        heatmap_yhat2.scale([0.29, 0.28, 0.28])
        heatmap_yhat2.move_to([0.96,0,0])
        heatmap_yhat2.set_opacity(0.5)
        heatmaps.add(heatmap_yhat2)

        canvas_x, canvas_y=latlong_to_canvas(xs[i][0], xs[i][1], label=ys[i], min_long=min_long, 
                                             max_long=max_long, min_lat=min_lat, max_lat=max_lat, berlin_adjust=[-0.01, 0.02, 0])
        training_point=m.Dot([canvas_x, canvas_y, 0], radius=0.012)
        if ys[i]==0.0: training_point.set_color('#00FFFF')
        elif ys[i]==1.0: training_point.set_color(YELLOW)
        elif ys[i]==2.0: training_point.set_color(GREEN)   

        self.set_view(0, 0, 0, (-0.61, 0.01, 0.0), 1.50)
        self.add(net_background, nums)
        self.wait()


        self.play(m.FadeIn(grad_regions))

        # I want to have gradients grow in, but need to keep moving - I can add an arrow in editing pointing to a lart/small grad
        # Hmm actualy I maybe just shrink and grow one and then add labels there?
        self.wait()
        self.play(grad_regions[0].animate.scale([1,0.1,1]), run_time=1.5)
        self.wait()
        self.play(grad_regions[0].animate.scale([1,10,1]), run_time=1.5)
        self.wait()

        # for g in grad_regions: g.scale([1, 0.02, 1])
        # self.add(grad_regions)
        # self.play(*[g.animate.scale([1,50,1]) for g in grad_regions], run_time=2)
        # self.wait()        

        europe_map=image_at_height(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])

        map_tick_overlays=m.SVGMobject(svg_path+'/map_tick_overlays_1.svg')[1:]
        map_tick_overlays.scale([0.965, 0.96, 0.965])
        map_tick_overlays.shift([-0.077, 0.0185, 0])
        self.wait()

        self.play(m.FadeIn(europe_map), m.FadeIn(map_tick_overlays), self.animate_view(0, 0, 0, (-0.04, 0.01, 0.0), 1.94), run_time=2)

        self.wait()
        self.add(training_point)
        self.wait()

        box=m.SurroundingRectangle(training_point, color=YELLOW, buff=0.025)
        self.play(m.Create(box))
        self.wait()
        self.play(m.FadeOut(box))

        self.add(heatmaps)
        heatmaps.set_opacity(0.0)
        self.remove(map_tick_overlays); self.add(map_tick_overlays)

        self.wait()
        self.play(heatmap_yhat1.animate.set_opacity(0.5))

        self.wait()
        self.play(heatmap_yhat2.animate.set_opacity(0.5))

        self.wait()
        self.play(heatmap_yhat3.animate.set_opacity(0.5))
        self.wait() 
        

        step_label=m.Text("Step=")
        step_label.set_color(CHILL_BROWN)
        step_label.scale(0.12)
        step_label.move_to([1.3, -0.85, 0])

        step_count=m.Text(str(i).zfill(3))
        step_count.set_color(CHILL_BROWN)
        step_count.scale(0.12)
        step_count.move_to([1.43, -0.85, 0])

        self.play(m.FadeIn(step_label), m.FadeIn(step_count))
        self.wait()

        for i in range(1, len(xs)):
            if i>0:
                self.remove(nums)
                self.remove(grad_regions)
                self.remove(heatmaps)
                self.remove(training_point) 
                self.remove(map_tick_overlays)
                if step_label is not None: 
                    self.remove(step_label,step_count)  
                # self.wait(0.1)       

            nums=get_numbers(i, xs, weights, logits, yhats)
            grad_regions=get_grad_regions(i, ys, yhats, grads)
            
            #Ok how do I do a cool surface again? Should i actually just try images first real quick?
            #Ok maybe images are fine for a bit - I'll switch to real surfaces when I need to

            heatmaps=m.Group()
            heatmap_yhat3=image_at_height(heatmap_path +'/'+str(i)+'_yhat_3.png')
            heatmap_yhat3.scale([0.29, 0.28, 0.28])
            heatmap_yhat3.move_to([0.96,0,0])
            heatmap_yhat3.set_opacity(0.5)
            heatmaps.add(heatmap_yhat3)

            heatmap_yhat1=image_at_height(heatmap_path +'/'+str(i)+'_yhat_1.png')
            heatmap_yhat1.scale([0.29, 0.28, 0.28])
            heatmap_yhat1.move_to([0.96,0,0])
            heatmap_yhat1.set_opacity(0.5)
            heatmaps.add(heatmap_yhat1)

            heatmap_yhat2=image_at_height(heatmap_path +'/'+str(i)+'_yhat_2.png')
            heatmap_yhat2.scale([0.29, 0.28, 0.28])
            heatmap_yhat2.move_to([0.96,0,0])
            heatmap_yhat2.set_opacity(0.5)
            heatmaps.add(heatmap_yhat2)

            #Ok, last piece of the puzzle here for a basic demo is to scatter the training points - let's go!
            #Ok this is kinda hacky/dumb - but let's just do little manual calibration to find corers on canvas
            #Then it's just an affine transform from boundaries - right?
            # bottom_left=Dot([0.38, -0.56, 0], radius=0.007)
            # self.add(bottom_left)
            # top_left=Dot([0.38, 0.56, 0], radius=0.007)
            # self.add(top_left)
            # bottom_right=Dot([1.54, -0.56, 0], radius=0.007)
            # self.add(bottom_right)
            # top_right=Dot([1.54, 0.56, 0], radius=0.007)
            # self.add(top_right)

            canvas_x, canvas_y=latlong_to_canvas(xs[i][0], xs[i][1], label=ys[i], min_long=min_long, 
                                             max_long=max_long, min_lat=min_lat, max_lat=max_lat, berlin_adjust=[-0.01, 0.02, 0], 
                                             paris_adjust=[-0.009, -0.002, 0], madrid_adjust=[-0.009, -0.002, 0])
            training_point=m.Dot([canvas_x, canvas_y, 0], radius=0.012)
            if ys[i]==0.0: training_point.set_color('#00FFFF')
            elif ys[i]==1.0: training_point.set_color(YELLOW)
            elif ys[i]==2.0: training_point.set_color(GREEN)   

            step_label=m.Text("Step=")
            step_label.set_color(CHILL_BROWN)
            step_label.scale(0.12)
            step_label.move_to([1.3, -0.85, 0])

            step_count=m.Text(str(i).zfill(3))
            step_count.set_color(CHILL_BROWN)
            step_count.scale(0.12)
            step_count.move_to([1.43, -0.85, 0])

            self.add(step_label,step_count) 
            self.add(nums)
            self.add(grad_regions)
            self.add(heatmaps)
            self.add(training_point)
            self.add(map_tick_overlays)
            self.wait(0.1)


        self.wait(20)
        pass



def get_numbers_b(i, xs, weights, logits, yhats):

    nums = m.VGroup()
    x = xs[i, -1]
    tx = math_tex(str(x) + r'^\circ')
    tx.scale(0.13)
    tx.move_to([-1.49, 0.02, 0])
    nums.add(tx)
    
    # Weights - using consistent formatting
    w = weights[i, :]
    tm1 = math_tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
    tm1.scale(0.12)
    tm1.move_to([-1.185, 0.54, 0])
    nums.add(tm1)
    
    tm2 = math_tex(format_number(w[1], total_chars=6)).set_color(YELLOW)
    tm2.scale(0.12)
    tm2.move_to([-1.185, 0.1, 0])
    nums.add(tm2)
    
    tm3 = math_tex(format_number(w[2], total_chars=6)).set_color(GREEN)
    tm3.scale(0.12)
    tm3.move_to([-1.185, -0.33, 0])
    nums.add(tm3)
    
    # Biases
    tb1 = math_tex(format_number(w[3], total_chars=6)).set_color('#00FFFF')
    tb1.scale(0.12)
    tb1.move_to([-1.185, 0.37, 0])
    nums.add(tb1)
    
    tb2 = math_tex(format_number(w[4], total_chars=6)).set_color(YELLOW)
    tb2.scale(0.12)
    tb2.move_to([-1.185, -0.07, 0])
    nums.add(tb2)
    
    tb3 = math_tex(format_number(w[5], total_chars=6)).set_color(GREEN)
    tb3.scale(0.12)
    tb3.move_to([-1.185, -0.51, 0])
    nums.add(tb3)
    
    # Logits
    tl1 = math_tex(format_number(logits[i, 0], total_chars=6)).set_color('#00FFFF')
    tl1.scale(0.16)
    tl1.move_to([-0.52, 0.37, 0])
    nums.add(tl1)
    
    tl2 = math_tex(format_number(logits[i, 1], total_chars=6)).set_color(YELLOW)
    tl2.scale(0.16)
    tl2.move_to([-0.52, 0.015, 0])
    nums.add(tl2)
    
    tl3 = math_tex(format_number(logits[i, 2], total_chars=6)).set_color(GREEN)
    tl3.scale(0.16)  
    tl3.move_to([-0.52, -0.335, 0])
    nums.add(tl3)
    
    # Predictions
    yhat1 = math_tex(format_number(yhats[i, 0], total_chars=6)).set_color('#00FFFF')
    yhat1.scale(0.16)
    yhat1.move_to([0.22, 0.37, 0])
    nums.add(yhat1)
    
    yhat2 = math_tex(format_number(yhats[i, 1], total_chars=6)).set_color(YELLOW)
    yhat2.scale(0.16)
    yhat2.move_to([0.22, 0.015, 0])
    nums.add(yhat2)
    
    yhat3 = math_tex(format_number(yhats[i, 2], total_chars=6)).set_color(GREEN)
    yhat3.scale(0.16)
    yhat3.move_to([0.22, -0.335, 0])
    nums.add(yhat3)

    return nums


class p46a(WelchScene):
    def construct(self):
        '''
        Ok so here i want to get the basic elements of the little linear models working
        Then I can working on smoothly transition from ball and stick to linaer -> shouldn't be terrible
        Before I do that I can work on some writing and decide if want to add any real time curves
        Finally, the most difficult/important thing is what I build on this next - 
        Brining 3 lines together, longitudes on bottom
        Then exapnd to planes -> that should be interesting
        Finally put over map, and morph smoothly to sofmax version - that should be interesting. 
        '''


        min_long=-9.8
        max_long=17.2
        min_lat=36.15 
        max_lat=54.7  

        data=np.load(data_path+'/cities_1d_5.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]

        #Snap back to first training step of p45 I think/

        net_background_0=m.SVGMobject(svg_path+'/p44_background_1.svg')
        
        i=0
        nums0=get_numbers(i, xs, weights, logits, yhats)
        grad_regions=get_grad_regions(i, ys, yhats, grads)
        heatmaps=m.Group()
        heatmap_yhat3=image_at_height(heatmap_path +'/'+str(i)+'_yhat_3.png')
        heatmap_yhat3.scale([0.29, 0.28, 0.28])
        heatmap_yhat3.move_to([0.96,0,0])
        heatmap_yhat3.set_opacity(0.5)
        heatmaps.add(heatmap_yhat3)

        heatmap_yhat1=image_at_height(heatmap_path +'/'+str(i)+'_yhat_1.png')
        heatmap_yhat1.scale([0.29, 0.28, 0.28])
        heatmap_yhat1.move_to([0.96,0,0])
        heatmap_yhat1.set_opacity(0.5)
        heatmaps.add(heatmap_yhat1)

        heatmap_yhat2=image_at_height(heatmap_path +'/'+str(i)+'_yhat_2.png')
        heatmap_yhat2.scale([0.29, 0.28, 0.28])
        heatmap_yhat2.move_to([0.96,0,0])
        heatmap_yhat2.set_opacity(0.5)
        heatmaps.add(heatmap_yhat2)

        canvas_x, canvas_y=latlong_to_canvas(xs[i][0], xs[i][1], label=ys[i], min_long=min_long, 
                                             max_long=max_long, min_lat=min_lat, max_lat=max_lat, berlin_adjust=[-0.01, 0.02, 0])
        training_point=m.Dot([canvas_x, canvas_y, 0], radius=0.012)
        if ys[i]==0.0: training_point.set_color('#00FFFF')
        elif ys[i]==1.0: training_point.set_color(YELLOW)
        elif ys[i]==2.0: training_point.set_color(GREEN)   

        europe_map=image_at_height(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])        

        map_tick_overlays=m.SVGMobject(svg_path+'/map_tick_overlays_1.svg')[1:]
        map_tick_overlays.scale([0.965, 0.96, 0.965])
        map_tick_overlays.shift([-0.077, 0.0185, 0])

        step_label=m.Text("Step=")
        step_label.set_color(CHILL_BROWN)
        step_label.scale(0.12)
        step_label.move_to([1.3, -0.85, 0])

        step_count=m.Text(str(i).zfill(3))
        step_count.set_color(CHILL_BROWN)
        step_count.scale(0.12)
        step_count.move_to([1.43, -0.85, 0])

        self.set_view(0, 0, 0, (-0.04, 0.01, 0.0), 1.94)
        self.add(net_background_0, nums0)
        self.add(grad_regions)
        self.add(europe_map)
        self.add(heatmaps)
        self.add(training_point)
        self.add(step_label, step_count)
        self.add(map_tick_overlays)
        self.wait()


        #Load up stuff for big transition
        net_background=m.SVGMobject(svg_path+'/p44_background_2.svg')[1:]
        # self.add(net_background)

        axes_1 = diagram_axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )
        axes_2=diagram_axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )
        axes_3=diagram_axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":3,
                "tip_config": {"width":0.02, "length":0.02}
                }
        )

        axes_1.move_to([-0.95, 0.44, 0])
        axes_2.move_to([-0.95, 0.0, 0])
        axes_3.move_to([-0.95, -0.44, 0])
        

        nums=get_numbers_b(i, xs, weights, logits, yhats)


        #End intial/transition setup
        self.wait()
        self.play(m.FadeOut(grad_regions))
        self.play(m.FadeOut(net_background_0), m.FadeIn(net_background),
                  *[m.ReplacementTransform(nums0[i], nums[i]) for i in range(len(nums))], run_time=2.5)
        self.add(axes_1, axes_2, axes_3)
        self.wait()

        def line_function_1(x): return weights[i,0] * x + weights[i,3]
        line_1 = axes_1.plot(line_function_1, color='#00FFFF', x_range=[-12, 12])
        arrow_tip_1 = get_arrow_tip(line_1, color='#00FFFF', scale=0.1)

        def line_function_2(x): return weights[i,1] * x + weights[i,4]
        line_2 = axes_2.plot(line_function_2, color=YELLOW, x_range=[-12, 12])
        arrow_tip_2 = get_arrow_tip(line_2, color=YELLOW, scale=0.1)

        def line_function_3(x): return weights[i,2] * x + weights[i,5]
        line_3 = axes_3.plot(line_function_3, color=GREEN, x_range=[-12, 12])
        arrow_tip_3 = get_arrow_tip(line_3, color=GREEN, scale=0.1)

        self.play(m.Create(m.VGroup(line_1, arrow_tip_1)))
        self.wait()

        self.play(m.Create(m.VGroup(line_2, arrow_tip_2)))
        self.wait()

        self.play(m.Create(m.VGroup(line_3, arrow_tip_3)))
        self.wait()

        for i in range(1, len(xs)):
            if i>0:
                self.remove(line_1, arrow_tip_1)
                self.remove(line_2, arrow_tip_2)
                self.remove(line_3, arrow_tip_3)
                self.remove(nums)
                self.remove(step_label,step_count)
                self.remove(heatmaps)
                self.remove(map_tick_overlays)

  
            nums=get_numbers_b(i, xs, weights, logits, yhats)

        
            def line_function_1(x): return weights[i,0] * x + weights[i,3]
            line_1 = axes_1.plot(line_function_1, color='#00FFFF', x_range=[-12, 12])
            arrow_tip_1 = get_arrow_tip(line_1, color='#00FFFF', scale=0.1)

            def line_function_2(x): return weights[i,1] * x + weights[i,4]
            line_2 = axes_2.plot(line_function_2, color=YELLOW, x_range=[-12, 12])
            arrow_tip_2 = get_arrow_tip(line_2, color=YELLOW, scale=0.1)

            def line_function_3(x): return weights[i,2] * x + weights[i,5]
            line_3 = axes_3.plot(line_function_3, color=GREEN, x_range=[-12, 12])
            arrow_tip_3 = get_arrow_tip(line_3, color=GREEN, scale=0.1)

            heatmaps=m.Group()
            heatmap_yhat3=image_at_height(heatmap_path +'/'+str(i)+'_yhat_3.png')
            heatmap_yhat3.scale([0.29, 0.28, 0.28])
            heatmap_yhat3.move_to([0.96,0,0])
            heatmap_yhat3.set_opacity(0.5)
            heatmaps.add(heatmap_yhat3)

            heatmap_yhat1=image_at_height(heatmap_path +'/'+str(i)+'_yhat_1.png')
            heatmap_yhat1.scale([0.29, 0.28, 0.28])
            heatmap_yhat1.move_to([0.96,0,0])
            heatmap_yhat1.set_opacity(0.5)
            heatmaps.add(heatmap_yhat1)

            heatmap_yhat2=image_at_height(heatmap_path +'/'+str(i)+'_yhat_2.png')
            heatmap_yhat2.scale([0.29, 0.28, 0.28])
            heatmap_yhat2.move_to([0.96,0,0])
            heatmap_yhat2.set_opacity(0.5)
            heatmaps.add(heatmap_yhat2)
            
            step_label=m.Text("Step=")
            step_label.set_color(CHILL_BROWN)
            step_label.scale(0.12)
            step_label.move_to([1.3, -0.85, 0])

            step_count=m.Text(str(i).zfill(3))
            step_count.set_color(CHILL_BROWN)
            step_count.scale(0.12)
            step_count.move_to([1.43, -0.85, 0])

            self.add(step_label,step_count) 
            self.add(axes_1, line_1, arrow_tip_1)
            self.add(axes_2, line_2, arrow_tip_2)
            self.add(axes_3, line_3, arrow_tip_3)
            self.add(nums)
            self.add(heatmaps)
            self.add(map_tick_overlays)
            self.wait(0.1)



        self.wait(20)
        pass


class p46b_v2(WelchScene):
    def construct(self):
        '''
        Ok starting with p45, I'll work on animating to shared p46 plot, and then start hacking on 3d. 
        '''
        data=np.load(data_path+'/cities_1d_5.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]


        net_background=m.SVGMobject(svg_path+'/p44_background_2.svg')
        
        europe_map=image_at_height(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])
        
        #Add 3d map early to layering works out.
        europe_map_2=image_at_height(svg_path +'/map_exports.00_00_01_13.Still001.png')
        europe_map_2.scale(0.28)
        europe_map_2.move_to([0.96,0,0])
        europe_map_2.scale(0.4)
        europe_map_2.move_to([-1.005, 0.25, -0.01]) #Not going to fine tune too much here until I have the final map. 
        self.add(europe_map_2)
        europe_map_2.set_opacity(0.0)


        map_tick_overlays=m.SVGMobject(svg_path+'/map_tick_overlays_1.svg')[1:]
        map_tick_overlays.scale([0.965, 0.96, 0.965])
        map_tick_overlays.shift([-0.077, 0.0185, 0])
 
        axes_1 = diagram_axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )
        axes_2=diagram_axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )
        axes_3=diagram_axes(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            width=0.32,
            height=0.32,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_1.move_to([-0.95, 0.44, 0])
        axes_2.move_to([-0.95, 0.0, 0])
        axes_3.move_to([-0.95, -0.44, 0])
        

        i=0
        nums=get_numbers_b(i, xs, weights, logits, yhats)
    
        def line_function_1(x): return weights[i,0] * x + weights[i,3]
        line_1 = axes_1.plot(line_function_1, stroke_width=3, color='#00FFFF', x_range=[-12, 12])
        arrow_tip_1 = get_arrow_tip(line_1, color='#00FFFF', scale=0.08)

        def line_function_2(x): return weights[i,1] * x + weights[i,4]
        line_2 = axes_2.plot(line_function_2, stroke_width=3, color=YELLOW, x_range=[-12, 12])
        arrow_tip_2 = get_arrow_tip(line_2, color=YELLOW, scale=0.08)

        def line_function_3(x): return weights[i,2] * x + weights[i,5]
        line_3 = axes_3.plot(line_function_3, stroke_width=3, color=GREEN, x_range=[-12, 12])
        arrow_tip_3 = get_arrow_tip(line_3, color=GREEN, scale=0.08)

        heatmaps=m.Group()
        heatmap_yhat3=image_at_height(heatmap_path +'/'+str(i)+'_yhat_3.png')
        heatmap_yhat3.scale([0.29, 0.28, 0.28])
        heatmap_yhat3.move_to([0.96,0,0])
        heatmap_yhat3.set_opacity(0.5)
        heatmaps.add(heatmap_yhat3)

        heatmap_yhat1=image_at_height(heatmap_path +'/'+str(i)+'_yhat_1.png')
        heatmap_yhat1.scale([0.29, 0.28, 0.28])
        heatmap_yhat1.move_to([0.96,0,0])
        heatmap_yhat1.set_opacity(0.5)
        heatmaps.add(heatmap_yhat1)

        heatmap_yhat2=image_at_height(heatmap_path +'/'+str(i)+'_yhat_2.png')
        heatmap_yhat2.scale([0.29, 0.28, 0.28])
        heatmap_yhat2.move_to([0.96,0,0])
        heatmap_yhat2.set_opacity(0.5)
        heatmaps.add(heatmap_yhat2)


        self.set_view(0, 0, 0, (-0.04, 0.01, 0.0), 1.94)
        self.add(net_background)
        self.add(europe_map)
        self.add(axes_1, axes_2, axes_3)


        self.add(axes_1, line_1, arrow_tip_1)
        self.add(axes_2, line_2, arrow_tip_2)
        self.add(axes_3, line_3, arrow_tip_3)
        self.add(nums)
        self.add(heatmaps)
        self.add(map_tick_overlays)
        self.wait()


        #Getting setup to merge plots
        top_plot_group=m.VGroup(axes_1, line_1, arrow_tip_1)
        middle_plot_group=m.VGroup(axes_2, line_2, arrow_tip_2)
        bottom_plot_group=m.VGroup(axes_3, line_3, arrow_tip_3)
        

        background_elements_to_remove=background_elements_to_remove = [31, 32, 33, 34, 35, 36, 45, 46, 47, 48, 49, 50, 82, 83, 84, 85, 86, 89, 90]
        background_elements_to_keep = [i for i in range(len(net_background)) if i not in background_elements_to_remove]
        self.wait()

        self.play(top_plot_group.animate.scale(1.5).move_to([-1.0, 0.025, 0]),
                  middle_plot_group.animate.scale(1.5).move_to([-1.0, 0.025, 0]),
                  bottom_plot_group.animate.scale(1.5).move_to([-1.0, 0.025, 0]),
                  *[m.FadeOut(net_background[o]) for o in background_elements_to_remove],
                  *[net_background[o].animate.set_opacity(0.5) for o in background_elements_to_keep],
                  m.FadeOut(nums[1:7]),
                  nums[0].animate.set_opacity(0.3), 
                  nums[7:].animate.set_opacity(0.3),
                  # arrow_tip_1.animate.scale(0.7),
                  # arrow_tip_2.animate.scale(0.7),
                  # arrow_tip_3.animate.scale(0.7),
                  self.animate_view(0, 0, 0, (-0.64, 0.0, 0.0), 1.14),
                  run_time=4.0
            )

        self.remove(axes_1, axes_3)
        self.remove(arrow_tip_2) #Occlusions
        self.add(arrow_tip_2)
        self.wait()



        #Ok so now we ad in overlays showing mapping
        # Ok here's where I first add the -10/+20 labels, these might get me into trouble later
        # let's see how it shakes out and then make a call. 
        overlays_1=m.SVGMobject(svg_path+'/p46_overlays_1.svg')
        overlays_2=m.SVGMobject(svg_path+'/p46_overlays_2.svg')
        overlays_1.scale(0.57)
        overlays_1.move_to([-0.64, 0.003, 0])

        self.wait()

        self.play(m.FadeIn(overlays_1[1:]), nums[0].animate.set_opacity(1.0), nums[7].animate.set_opacity(1.0))
        self.wait()

        box=m.SurroundingRectangle(nums[-3], color=YELLOW, buff=0.025)
        self.play(nums[-3].animate.set_opacity(1.0), m.Create(box))
        self.wait()
        self.play(nums[-3].animate.set_opacity(0.3), m.FadeOut(box))
        self.wait(0)


        overlays_2.scale(0.30)
        overlays_2.move_to([-1.02, 0.025, 0])
        madrid_label=overlays_2[1:18] #Nudge this bad boi?
        madrid_label.shift([0.015, 0, 0])
        self.wait()

        #Zoom in and run training just on little plot
        self.play(self.animate_view(0, 0, 0, (-1.02, 0.02, 0.0), 0.62),
                  *[net_background[o].animate.set_opacity(0.0) for o in background_elements_to_keep],
                  nums[0].animate.set_opacity(0.0), nums[7:].animate.set_opacity(0.0), 
                  overlays_1[1:].animate.set_opacity(0.0), run_time=3.0
                 )
        self.add(overlays_2[1:])
        self.wait()
        
        #Labels are a bit innacurate here, gotta decide if I want to fix or not worry about it
        # Does seem like I should just nudge Madrid to the right a bit? Shouldn't be that bad...
        # I could actually maybe even nudge it in manim...

        # Ok, now I need to run training from this zoomed in spot, and then go back to the beginning I think and 
        # go to 3d! 
        # I want to include step count here for sure. 
        step_label=m.Text("Step=")
        step_label.set_color(CHILL_BROWN)
        step_label.scale(0.04)
        step_label.move_to([-0.58, -0.25, 0])

        step_count=m.Text(str(i).zfill(3))
        step_count.set_color(CHILL_BROWN)
        step_count.scale(0.04)
        step_count.next_to(step_label, m.RIGHT, buff=0.003)
        self.play(m.FadeIn(step_label), m.FadeIn(step_count))

        for i in range(1, len(xs)): #TO DO -> CHANGE STARTING NUMBER TO 1 TO ACTUALLY PLAY ANIMATION
            if i>0:
                self.remove(line_1, arrow_tip_1)
                self.remove(line_2, arrow_tip_2)
                self.remove(line_3, arrow_tip_3)
                self.remove(step_label,step_count)

            nums=get_numbers_b(i, xs, weights, logits, yhats)
        
            def line_function_1(x): return weights[i,0] * x + weights[i,3]
            line_1 = axes_1.plot(line_function_1, color='#00FFFF', x_range=[-12, 12])
            arrow_tip_1 = get_arrow_tip(line_1, color='#00FFFF', scale=0.1)

            def line_function_2(x): return weights[i,1] * x + weights[i,4]
            line_2 = axes_2.plot(line_function_2, color=YELLOW, x_range=[-12, 12])
            arrow_tip_2 = get_arrow_tip(line_2, color=YELLOW, scale=0.1)

            def line_function_3(x): return weights[i,2] * x + weights[i,5]
            line_3 = axes_3.plot(line_function_3, color=GREEN, x_range=[-12, 12])
            arrow_tip_3 = get_arrow_tip(line_3, color=GREEN, scale=0.1)

            heatmaps.add(heatmap_yhat2)
            
            step_label=m.Text("Step=")
            step_label.set_color(CHILL_BROWN)
            step_label.scale(0.04)
            step_label.move_to([-0.58, -0.25, 0])

            step_count=m.Text(str(i).zfill(3))
            step_count.set_color(CHILL_BROWN)
            step_count.scale(0.04)
            step_count.next_to(step_label, m.RIGHT, buff=0.003)

            self.add(step_label,step_count) 
            self.add(line_1, arrow_tip_1)
            self.add(line_2, arrow_tip_2)
            self.add(line_3, arrow_tip_3)
            self.wait(0.1)
        

        self.wait()

        #Hmm i need ot make the lines longer...

        def line_function_1(x): return weights[i,0] * x + weights[i,3]
        line_1_long = axes_1.plot(line_function_1, color='#00FFFF', x_range=[-14.5, 14.5])
        arrow_tip_1_long = get_arrow_tip(line_1_long, color='#00FFFF', scale=0.1)

        def line_function_2(x): return weights[i,1] * x + weights[i,4]
        line_2_long = axes_2.plot(line_function_2, color=YELLOW, x_range=[-14.5, 14.5])
        arrow_tip_2_long = get_arrow_tip(line_2_long, color=YELLOW, scale=0.1)

        def line_function_3(x): return weights[i,2] * x + weights[i,5]
        line_3_long = axes_3.plot(line_function_3, color=GREEN, x_range=[-14.5, 14.5])
        arrow_tip_3_long = get_arrow_tip(line_3_long, color=GREEN, scale=0.1)



        #Can i jump to 3d cleanly?
        # self.animate_view(0, 0, 0, (-1.02, 0.02, 0.0), 0.62)

        # self.play(ReplacementTransform(line_1, line_1_long))
        self.wait()
        self.remove(europe_map, heatmaps, map_tick_overlays, step_label, step_count)
        stuff_to_rotate=m.VGroup(axes_2, line_1_long, line_2_long, line_3_long, arrow_tip_1_long, arrow_tip_2_long, arrow_tip_3_long,
                              line_1, line_2, line_3, arrow_tip_1, arrow_tip_2, arrow_tip_3)
        stuff_to_rotate.rotate(90*m.DEGREES, [1, 0, 0])
        overlays_2.rotate(90*m.DEGREES, [1, 0, 0])
        self.set_view(0, 90, 0, (-1.02, 0.02, 0.0), 0.62)
        self.wait()

        #Ok so there is a little jump here -> I'm going to try to fix/hide it in premiere to save time. 
        europe_map_2.set_opacity(1.0)
        
        madrid_label=overlays_2[1:18]
        # madrid_center=madrid_label.get_center()
        berlin_label=overlays_2[33:50]
        paris_label=overlays_2[18:33]


        plane_1_zero = create_plane_from_line_endpoints(line_1_long, '#00FFFF', y_extension=0.01)
        plane_1_full = create_plane_from_line_endpoints(line_1_long, '#00FFFF', y_extension=0.5)
        plane_1_zero.set_opacity(0.3)
        plane_1_full.set_opacity(0.3)

        plane_2_zero = create_plane_from_line_endpoints(line_2_long, YELLOW, y_extension=0.01)
        plane_2_full = create_plane_from_line_endpoints(line_2_long, YELLOW, y_extension=0.5)
        plane_2_zero.set_opacity(0.3)
        plane_2_full.set_opacity(0.3)

        plane_3_zero = create_plane_from_line_endpoints(line_3_long, GREEN, y_extension=0.01)
        plane_3_full = create_plane_from_line_endpoints(line_3_long, GREEN, y_extension=0.5)
        plane_3_zero.set_opacity(0.3)
        plane_3_full.set_opacity(0.3)

        self.wait()
        self.play(self.animate_view(-30, 59, 0, (-1.02, 0.04, -0.0), 0.62),
                        madrid_label.animate.move_to([-1.065,  0.12     , -0.06620278]),
                        paris_label.animate.move_to([-0.97,  0.295     , -0.06596944]),
                        berlin_label.animate.move_to([-0.805,  0.38     , -0.06464722]),
                    run_time=3.0
        )
        self.wait()

        #Ok the dope thing here would be planes coming out one at a time during a continuous camera move. 
        self.play(self.animate_view(18, 57, 0, (-0.95, 0.08, 0.05), 0.62),
                 m.Transform(plane_1_zero, plane_1_full),
                 m.Transform(plane_2_zero, plane_2_full),
                 m.Transform(plane_3_zero, plane_3_full),
                 run_time=3.0)
        self.wait()


        def softmax_function(x):
            # Compute softmax for this specific x value
            logit_1 = weights[i][0] * x + weights[i][3]
            logit_2 = weights[i][1] * x + weights[i][4] 
            logit_3 = weights[i][2] * x + weights[i][5]

            logits = np.array([logit_1, logit_2, logit_3])
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            return softmax_viz_scale*probabilities[neuron_index]

        
        softmax_viz_scale=5.0
        neuron_index=0
        softmax_curve_1 = axes_2.plot(softmax_function, x_range=(-14.5, 14.5), color='#00FFFF', stroke_width=3)
        neuron_index=1
        softmax_curve_2 = axes_2.plot(softmax_function, x_range=(-14.5, 14.5), color=YELLOW, stroke_width=3)
        neuron_index=2
        softmax_curve_3 = axes_2.plot(softmax_function, x_range=(-14.5, 14.5), color=GREEN, stroke_width=3)

        # self.add(softmax_curve_1) 

        self.wait()
        softmax_curve_1.set_fill(opacity=0)
        softmax_curve_1.set_stroke(color='#00FFFF', width=3)
        softmax_curve_2.set_fill(opacity=0)
        softmax_curve_2.set_stroke(color=YELLOW, width=3)
        softmax_curve_3.set_fill(opacity=0)
        softmax_curve_3.set_stroke(color=GREEN, width=3)
        
        # Make sure original line also has no fill
        line_1_long.set_fill(opacity=0)
        line_1_long.set_stroke(color='#00FFFF', width=3)
        line_2_long.set_fill(opacity=0)
        line_2_long.set_stroke(color=YELLOW, width=3)
        line_3_long.set_fill(opacity=0)
        line_3_long.set_stroke(color=GREEN, width=3)

        line_1.set_fill(opacity=0)
        line_1.set_stroke(color='#00FFFF', width=3)
        line_2.set_fill(opacity=0)
        line_2.set_stroke(color=YELLOW, width=3)
        line_3.set_fill(opacity=0)
        line_3.set_stroke(color=GREEN, width=3)
        
        softmax_surface_1 = create_surface_from_curve_simple(curve=softmax_curve_1, y_extension=0.5)   # Match your plane y_extension color='#00FFFF'
        softmax_surface_2 = create_surface_from_curve_simple(curve=softmax_curve_2, y_extension=0.5)
        softmax_surface_3 = create_surface_from_curve_simple(curve=softmax_curve_3, y_extension=0.5)


        matching_plane_1, matching_surface_1 = create_matching_plane_and_surface(line_1_long, softmax_curve_1, y_extension=0.5, color='#00FFFF')
        matching_plane_2, matching_surface_2 = create_matching_plane_and_surface(line_2_long, softmax_curve_2, y_extension=0.5, color=YELLOW)
        matching_plane_3, matching_surface_3 = create_matching_plane_and_surface(line_3_long, softmax_curve_3, y_extension=0.5, color=GREEN)
        
        self.wait()


        self.remove(plane_1_zero, plane_2_zero, plane_3_zero)
        self.add(matching_plane_1, matching_plane_2, matching_plane_3)
        self.remove(arrow_tip_1, arrow_tip_2, arrow_tip_3)
        self.play(m.Transform(matching_plane_1, matching_surface_1),
                  m.ReplacementTransform(line_1, softmax_curve_1),
                  m.Transform(matching_plane_2, matching_surface_2),
                  m.ReplacementTransform(line_2, softmax_curve_2),
                  m.Transform(matching_plane_3, matching_surface_3),
                  m.ReplacementTransform(line_3, softmax_curve_3),
                  run_time=6)  
        matching_plane_3.shift([0,0,-0.0001]) #Removes a visual artifact 
        self.wait()

        # self.remove(matching_plane_3)
        # self.add(matching_plane_3)


        # self.remove(matching_plane_1); self.add(matching_plane_1)
        # self.remove(matching_surface_1); #self.add(matching_plane_1)
        # self.remove(matching_surface_2)
        # self.remove(matching_surface_3)
        # self.remove(matching_plane_2); self.add(matching_plane_2)
        # self.remove(matching_plane_3); #self.add(matching_surface_3)

        


        self.play(self.animate_view(0, 0, 0, (-1.03, 0.24, 0.1), 0.62), run_time=7)
        self.wait()

        #Looking good - just seeing some green jumpiness in one of the surfaces on the move? Let me do the real render and see if it persists. 




        self.wait(20)
        pass





