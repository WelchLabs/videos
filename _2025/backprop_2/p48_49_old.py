from welch_paths import welch_path
import manimgx as m
from welch_geometry import diagram_axes_3d
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
heatmap_path=welch_path('backprop2/graphics/to_manim/may_28_2')


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



def latlong_to_canvas(lat, long, 
                      map_min_x=0.38, map_max_x=1.54,
                      map_min_y=-0.56, map_max_y=0.56,
                      min_long=-7.0, max_long=18.0,
                      min_lat=36.0, max_lat=56.0):
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
    
    return x, y



def get_dem_numbers_3d(i, xs, weights, logits, yhats):
    x1 = xs[i, 0]  # First input feature
    x2 = xs[i, 1]  # Second input feature
    
    nums = m.VGroup()
    
    # First input value
    tx1 = math_tex(str(x1) + r'^\circ')
    tx1.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tx1.move_to([-1.53, 0, 0.155])  
    nums.add(tx1)
    
    # Second input value
    tx2 = math_tex(str(x2) + r'^\circ')
    tx2.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tx2.move_to([-1.52, 0, -0.19])  
    nums.add(tx2)

    #   Neuron 1 weights (cyan)
    w = weights[i, :]
    tm1_1 = math_tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
    tm1_1.scale(0.12)
    tm1_1.rotate(90*m.DEGREES, [1, 0, 0])
    tm1_1.move_to([-1.04, 0, 0.85])
    nums.add(tm1_1)
    
    tm1_2 = math_tex(format_number(w[1], total_chars=6)).set_color('#00FFFF')
    tm1_2.scale(0.12)
    tm1_2.rotate(90*m.DEGREES, [1, 0, 0])
    tm1_2.move_to([-1.04, 0, 0.72])
    nums.add(tm1_2)       

    tb1 = math_tex(format_number(w[8], total_chars=6)).set_color('#00FFFF')
    tb1.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tb1.move_to([-1.04,0,  0.59])
    nums.add(tb1)

    tm2_1 = math_tex(format_number(w[2], total_chars=6)).set_color(YELLOW)
    tm2_1.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tm2_1.move_to([-1.04,0,  0.38])
    nums.add(tm2_1)
    
    tm2_2 = math_tex(format_number(w[3], total_chars=6)).set_color(YELLOW)
    tm2_2.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tm2_2.move_to([-1.04,0,  0.25])
    nums.add(tm2_2)       

    tb2 = math_tex(format_number(w[9], total_chars=6)).set_color(YELLOW)
    tb2.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tb2.move_to([-1.04,0,  0.12])
    nums.add(tb2)

    tm3_1 = math_tex(format_number(w[4], total_chars=6)).set_color(GREEN)
    tm3_1.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tm3_1.move_to([-1.04, 0, -0.08])
    nums.add(tm3_1)
    
    t3_2 = math_tex(format_number(w[5], total_chars=6)).set_color(GREEN)
    t3_2.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    t3_2.move_to([-1.04, 0, -0.21])
    nums.add(t3_2)       

    tb3 = math_tex(format_number(w[10], total_chars=6)).set_color(GREEN)
    tb3.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tb3.move_to([-1.04, 0, -0.34])
    nums.add(tb3)

    tm4_1 = math_tex(format_number(w[6], total_chars=6)).set_color('#FF00FF')
    tm4_1.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tm4_1.move_to([-1.04, 0, -0.54])
    nums.add(tm4_1)
    
    t4_2 = math_tex(format_number(w[7], total_chars=6)).set_color('#FF00FF')
    t4_2.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    t4_2.move_to([-1.04, 0, -0.68])
    nums.add(t4_2)       

    tb4 = math_tex(format_number(w[11], total_chars=6)).set_color('#FF00FF')
    tb4.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
    tb4.move_to([-1.04, 0, -0.82])
    nums.add(tb4)


    # Logits
    tl1 = math_tex(format_number(logits[i, 0], total_chars=6)).set_color('#00FFFF')
    tl1.scale(0.14).rotate(90*m.DEGREES, [1, 0, 0])
    tl1.move_to([-0.49, 0, 0.54])
    nums.add(tl1)
    
    tl2 = math_tex(format_number(logits[i, 1], total_chars=6)).set_color(YELLOW)
    tl2.scale(0.14).rotate(90*m.DEGREES, [1, 0, 0])
    tl2.move_to([-0.48, 0, 0.18])
    nums.add(tl2)
    
    tl3 = math_tex(format_number(logits[i, 2], total_chars=6)).set_color(GREEN)
    tl3.scale(0.14).rotate(90*m.DEGREES, [1, 0, 0])
    tl3.move_to([-0.49, -0, -0.17]) #tl3.move_to([-0.49, -0.17, 0])
    nums.add(tl3)

    tl4 = math_tex(format_number(logits[i, 3], total_chars=6)).set_color("#FF00FF")
    tl4.scale(0.14).rotate(90*m.DEGREES, [1, 0, 0])
    tl4.move_to([-0.48, 0, -0.5])
    nums.add(tl4)

    #Predictions
    yhat1 = math_tex(format_number(yhats[i, 0], total_chars=6)).set_color('#00FFFF')
    yhat1.scale(0.18).rotate(90*m.DEGREES, [1, 0, 0])
    yhat1.move_to([0.18, 0, 0.36])
    nums.add(yhat1)
    
    yhat2 = math_tex(format_number(yhats[i, 1], total_chars=6)).set_color(YELLOW)
    yhat2.scale(0.18).rotate(90*m.DEGREES, [1, 0, 0])
    yhat2.move_to([0.18, 0, 0.12])
    nums.add(yhat2)
    
    yhat3 = math_tex(format_number(yhats[i, 2], total_chars=6)).set_color(GREEN)
    yhat3.scale(0.18).rotate(90*m.DEGREES, [1, 0, 0])
    yhat3.move_to([0.18, 0, -0.12])
    nums.add(yhat3)

    yhat4 = math_tex(format_number(yhats[i, 3], total_chars=6)).set_color('#FF00FF')
    yhat4.scale(0.18).rotate(90*m.DEGREES, [1, 0, 0])
    yhat4.move_to([0.18, 0, -0.35])
    nums.add(yhat4)

    return nums

def get_dem_numbers(i, xs, weights, logits, yhats):
    x1 = xs[i, 0]  # First input feature
    x2 = xs[i, 1]  # Second input feature
    
    nums = m.VGroup()
    
    # First input value
    tx1 = math_tex(str(x1) + r'^\circ')
    tx1.scale(0.12)
    tx1.move_to([-1.53, 0.155, 0])  
    nums.add(tx1)
    
    # Second input value
    tx2 = math_tex(str(x2) + r'^\circ')
    tx2.scale(0.12)
    tx2.move_to([-1.52, -0.19, 0])  
    nums.add(tx2)

    #   Neuron 1 weights (cyan)
    w = weights[i, :]
    tm1_1 = math_tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
    tm1_1.scale(0.12)
    tm1_1.move_to([-1.04, 0.85, 0])
    nums.add(tm1_1)
    
    tm1_2 = math_tex(format_number(w[1], total_chars=6)).set_color('#00FFFF')
    tm1_2.scale(0.12)
    tm1_2.move_to([-1.04, 0.72, 0])
    nums.add(tm1_2)       

    tb1 = math_tex(format_number(w[8], total_chars=6)).set_color('#00FFFF')
    tb1.scale(0.12)
    tb1.move_to([-1.04, 0.59, 0])
    nums.add(tb1)

    tm2_1 = math_tex(format_number(w[2], total_chars=6)).set_color(YELLOW)
    tm2_1.scale(0.12)
    tm2_1.move_to([-1.04, 0.38, 0])
    nums.add(tm2_1)
    
    tm2_2 = math_tex(format_number(w[3], total_chars=6)).set_color(YELLOW)
    tm2_2.scale(0.12)
    tm2_2.move_to([-1.04, 0.25, 0])
    nums.add(tm2_2)       

    tb2 = math_tex(format_number(w[9], total_chars=6)).set_color(YELLOW)
    tb2.scale(0.12)
    tb2.move_to([-1.04, 0.12, 0])
    nums.add(tb2)

    tm3_1 = math_tex(format_number(w[4], total_chars=6)).set_color(GREEN)
    tm3_1.scale(0.12)
    tm3_1.move_to([-1.04, -0.08, 0])
    nums.add(tm3_1)
    
    t3_2 = math_tex(format_number(w[5], total_chars=6)).set_color(GREEN)
    t3_2.scale(0.12)
    t3_2.move_to([-1.04, -0.21, 0])
    nums.add(t3_2)       

    tb3 = math_tex(format_number(w[10], total_chars=6)).set_color(GREEN)
    tb3.scale(0.12)
    tb3.move_to([-1.04, -0.34, 0])
    nums.add(tb3)

    tm4_1 = math_tex(format_number(w[6], total_chars=6)).set_color('#FF00FF')
    tm4_1.scale(0.12)
    tm4_1.move_to([-1.04, -0.54, 0])
    nums.add(tm4_1)
    
    t4_2 = math_tex(format_number(w[7], total_chars=6)).set_color('#FF00FF')
    t4_2.scale(0.12)
    t4_2.move_to([-1.04, -0.68, 0])
    nums.add(t4_2)       

    tb4 = math_tex(format_number(w[11], total_chars=6)).set_color('#FF00FF')
    tb4.scale(0.12)
    tb4.move_to([-1.04, -0.82, 0])
    nums.add(tb4)


    # Logits
    tl1 = math_tex(format_number(logits[i, 0], total_chars=6)).set_color('#00FFFF')
    tl1.scale(0.14)
    tl1.move_to([-0.49, 0.54, 0])
    nums.add(tl1)
    
    tl2 = math_tex(format_number(logits[i, 1], total_chars=6)).set_color(YELLOW)
    tl2.scale(0.14)
    tl2.move_to([-0.48, 0.18, 0])
    nums.add(tl2)
    
    tl3 = math_tex(format_number(logits[i, 2], total_chars=6)).set_color(GREEN)
    tl3.scale(0.14)  
    tl3.move_to([-0.49, -0.17, 0])
    nums.add(tl3)

    tl4 = math_tex(format_number(logits[i, 3], total_chars=6)).set_color("#FF00FF")
    tl4.scale(0.14)  
    tl4.move_to([-0.48, -0.5, 0])
    nums.add(tl4)

    #Predictions
    yhat1 = math_tex(format_number(yhats[i, 0], total_chars=6)).set_color('#00FFFF')
    yhat1.scale(0.18)
    yhat1.move_to([0.18, 0.36, 0])
    nums.add(yhat1)
    
    yhat2 = math_tex(format_number(yhats[i, 1], total_chars=6)).set_color(YELLOW)
    yhat2.scale(0.18)
    yhat2.move_to([0.18, 0.12, 0])
    nums.add(yhat2)
    
    yhat3 = math_tex(format_number(yhats[i, 2], total_chars=6)).set_color(GREEN)
    yhat3.scale(0.18)
    yhat3.move_to([0.18, -0.12, 0])
    nums.add(yhat3)

    yhat4 = math_tex(format_number(yhats[i, 3], total_chars=6)).set_color('#FF00FF')
    yhat4.scale(0.18)
    yhat4.move_to([0.18, -0.35, 0])
    nums.add(yhat4)

    return nums

class LinearPlane(SampledSurface):
    """A plane defined by z = m1*x1 + m2*x2 + b"""
    def __init__(self, axes, m1=0.5, m2=0.3, b=1.0, vertical_viz_scale=0.5, **kwargs):
        self.axes = axes
        self.m1 = m1
        self.m2 = m2 
        self.b = b
        self.vertical_viz_scale=vertical_viz_scale
        super().__init__(
            # u_range=(-12, 12),
            # v_range=(-12, 12),
            u_range=(-6, 11),
            v_range=(-8, 4),
            resolution=(64, 64), #Looks nice at 256, but is slow, maybe crank for final
            color='#00FFFF',
            **kwargs
        )
    
    def uv_func(self, u, v):
        # u maps to x1, v maps to x2, compute z = m1*x1 + m2*x2 + b
        x1 = u
        x2 = v
        z = self.vertical_viz_scale*(self.m1 * x1 + self.m2 * x2 + self.b)
        # Transform to axes coordinate system
        return self.axes.c2p(x1, x2, z)



class p46_sketch_3(WelchScene):
    def construct(self):
        '''
        Ok now let's try to everything is Psuedo3d, so I can do the "bring everything together in 3D and pan around" deal 
        '''
        data=np.load(data_path+'/cities_2d_2.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:15]
        grads=data[:,15:27]
        logits=data[:,27:31]
        yhats=data[:, 31:]


        net_background=m.SVGMobject(svg_path+'/p_48_background_1.svg')
        
        # self.set_view(0, 0, 0, (-0.03, 0.01, 0.0), 2.01)
        europe_map=image_at_height(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])

        # self.add(net_background)
        # self.add(europe_map)

        axes_1 = diagram_axes_3d(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-6, 11, 1],
            y_range=[-8, 4, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_2 = diagram_axes_3d(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-6, 11, 1],
            y_range=[-8, 4, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_3 = diagram_axes_3d(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-6, 11, 1],
            y_range=[-8, 4, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_4 = diagram_axes_3d(
            # x_range=[-15, 15, 1],
            # y_range=[-15, 15, 1],
            x_range=[-6, 11, 1],
            y_range=[-8, 4, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )


        # stuff_to_rotate_once=Group(net_background, europe_map, axes_1) #, axes_1, axes_2, axes_3, axes_4)
        net_background.rotate(90*m.DEGREES, [1, 0, 0])
        europe_map.rotate(90*m.DEGREES, [1, 0, 0])
        self.add(net_background, europe_map)
        self.set_view(0, 90, 0, (-0.02, 0.01, -0.0), 1.98)
        # self.wait()

        axes_1.move_to([-0.80, 0, 0.7])
        # axes_1.rotate(-90*DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        # axes_1.rotate(-90*DEGREES, [0,0,1]) #Twist around vertical

        axes_2.move_to([-0.80, 0, 0.24])
        # axes_2.rotate(-90*DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        # axes_2.rotate(-90*DEGREES, [0,0,1]) #Twist around vertical
        
        axes_3.move_to([-0.80, 0, -0.22])
        # axes_3.rotate(-90*DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        # axes_3.rotate(-90*DEGREES, [0,0,1]) #Twist around vertical

        axes_4.move_to([-0.80, 0,  -0.7])
        # axes_4.rotate(-90*DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        # axes_4.rotate(-90*DEGREES, [0,0,1]) #Twist around vertical

        self.add(axes_1, axes_2, axes_3, axes_4)
        self.wait()

        vertical_viz_scale=0.4

        i=450

        # for i in range(len(xs)):
        #     if i>0:
        #         self.remove(nums)
        #         self.remove(plane_1, plane_2, plane_3, plane_4)
        #         # self.remove(grad_regions)
        #         self.remove(heatmaps)
        #         self.remove(training_point) 
        #         self.remove(step_label,step_count)  


        # nums=get_dem_numbers(i, xs, weights, logits, yhats)
        nums=get_dem_numbers_3d(i, xs, weights, logits, yhats)


        plane_1=LinearPlane(axes_1, weights[i,1], weights[i,0], weights[i,8], vertical_viz_scale=vertical_viz_scale)
        plane_1.set_opacity(0.6)
        plane_1.set_color('#00FFFF')

        plane_2=LinearPlane(axes_2, weights[i,3], weights[i,2], weights[i,9], vertical_viz_scale=vertical_viz_scale)
        plane_2.set_opacity(0.6)
        plane_2.set_color(YELLOW)

        plane_3=LinearPlane(axes_3, weights[i,5], weights[i,4], weights[i,10], vertical_viz_scale=vertical_viz_scale)
        plane_3.set_opacity(0.6)
        plane_3.set_color(GREEN)

        plane_4=LinearPlane(axes_4, weights[i,7], weights[i,6], weights[i,11], vertical_viz_scale=vertical_viz_scale)
        plane_4.set_opacity(0.6)
        plane_4.set_color('#FF00FF')

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

        heatmap_yhat4=image_at_height(heatmap_path +'/'+str(i)+'_yhat_4.png')
        heatmap_yhat4.scale([0.29, 0.28, 0.28])
        heatmap_yhat4.move_to([0.96,0,0])
        heatmap_yhat4.set_opacity(0.5)
        heatmaps.add(heatmap_yhat4)

        heatmaps.rotate(90*m.DEGREES, [1, 0, 0])

        canvas_x, canvas_y=latlong_to_canvas(xs[i][0], xs[i][1])
        training_point=m.Dot([canvas_x, 0, canvas_y], radius=0.012)
        if ys[i]==0.0: training_point.set_color('#00FFFF')
        elif ys[i]==1.0: training_point.set_color(YELLOW)
        elif ys[i]==2.0: training_point.set_color(GREEN)   
        elif ys[i]==3.0: training_point.set_color('#FF00FF')   
        training_point.rotate(90*m.DEGREES, [1, 0, 0])

        step_label=m.Text("Step=")
        step_label.set_color(CHILL_BROWN)
        step_label.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
        step_label.move_to([1.3, 0, -0.85])

        step_count=m.Text(str(i).zfill(3))
        step_count.set_color(CHILL_BROWN)
        step_count.scale(0.12).rotate(90*m.DEGREES, [1, 0, 0])
        step_count.move_to([1.43, 0, -0.85])

        self.add(plane_1, plane_2, plane_3, plane_4)
        self.add(nums)
        self.add(step_label,step_count) 
        self.add(heatmaps)
        self.add(training_point)

        self.wait()

        #Alright fancy 3d scene time, let's to! First let me manually remove everything that I'm going to fade out. 
        self.remove(nums)
        self.remove(net_background)
        self.remove(heatmaps)
        self.remove(training_point)
        self.remove(step_label,step_count) 

        # Ok, now for the animation -> probalby makes sense to scale and move the 3d axes at the same time? 
        # If that's funky I can try a camera move
        ap1=m.Group(axes_1, plane_1)
        ap2=m.Group(axes_2, plane_2)
        ap3=m.Group(axes_3, plane_3)
        ap4=m.Group(axes_4, plane_4)

        # ap1.move_to([0,0,0])
        # ap1.scale(5.5)

        # europe_map.rotate(-90*DEGREES, [1,0,0])
        # europe_map.move_to([-0.05,-0.05,0.00])

        self.wait()
        self.play(ap1.animate.scale(5.5).move_to([0,0,0]).set_opacity(0.4),
                    ap2.animate.scale(5.5).move_to([0,0,0]).set_opacity(0.4),
                    ap3.animate.scale(5.5).move_to([0,0,0]).set_opacity(0.4),
                    ap4.animate.scale(5.5).move_to([0,0,0]).set_opacity(0.4),
                    europe_map.animate.rotate(-90*m.DEGREES, [1,0,0]).move_to([-0.05,-0.05,0.00]),
                    self.animate_view(22, 35, 0, (0.04, -0.02, -0.04), 1.98),
                    run_time=7.0
            )
        self.remove(axes_2, axes_3, axes_4)
        self.wait()

        ## Ok this took some noodling, but the goemetry of the planes now matches the heatmap!
        ## I need to tweak spacing etc once i swap in the map, and do a nice pan, an dmaybe
        ## highlight one plane at a time with opacities, but proof of concept looks good!


        self.wait()








        self.wait()
        pass







class p46_sketch_2(WelchScene):
    def construct(self):
        '''
        Ok starting with p45, I'll work on animating to shared p46 plot, and then start hacking on 3d. 
        '''
        data=np.load(data_path+'/cities_2d_1.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:15]
        grads=data[:,15:27]
        logits=data[:,27:31]
        yhats=data[:, 31:]


        net_background=m.SVGMobject(svg_path+'/p_48_background_1.svg')
        self.add(net_background)

        self.set_view(0, 0, 0, (-0.03, 0.01, 0.0), 2.01)
        europe_map=image_at_height(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])
        self.add(europe_map)

        axes_1 = diagram_axes_3d(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_2 = diagram_axes_3d(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_3 = diagram_axes_3d(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_4 = diagram_axes_3d(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )


        axes_1.move_to([-0.80, 0.7, 0])
        axes_1.rotate(-90*m.DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        axes_1.rotate(-30*m.DEGREES, [0,1,0]) #Twist around vertical

        axes_2.move_to([-0.80, 0.24, 0])
        axes_2.rotate(-90*m.DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        axes_2.rotate(-30*m.DEGREES, [0,1,0]) #Twist around vertical

        axes_3.move_to([-0.80, -0.22, 0])
        axes_3.rotate(-90*m.DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        axes_3.rotate(-30*m.DEGREES, [0,1,0]) #Twist around vertical

        axes_4.move_to([-0.80, -0.7, 0])
        axes_4.rotate(-90*m.DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        axes_4.rotate(-30*m.DEGREES, [0,1,0]) #Twist around vertical

        self.add(axes_1, axes_2, axes_3, axes_4)
        vertical_viz_scale=0.4

        previous = []
        for i in range(len(xs)):
            self.remove(*previous)

            nums=get_dem_numbers(i, xs, weights, logits, yhats)

            plane_1=LinearPlane(axes_1, weights[i,0], weights[i,1], weights[i,8], vertical_viz_scale=vertical_viz_scale)
            plane_1.set_opacity(0.6)
            plane_1.set_color('#00FFFF')

            plane_2=LinearPlane(axes_2, weights[i,2], weights[i,3], weights[i,9], vertical_viz_scale=vertical_viz_scale)
            plane_2.set_opacity(0.6)
            plane_2.set_color(YELLOW)

            plane_3=LinearPlane(axes_3, weights[i,4], weights[i,5], weights[i,10], vertical_viz_scale=vertical_viz_scale)
            plane_3.set_opacity(0.6)
            plane_3.set_color(GREEN)

            plane_4=LinearPlane(axes_4, weights[i,6], weights[i,7], weights[i,11], vertical_viz_scale=vertical_viz_scale)
            plane_4.set_opacity(0.6)
            plane_4.set_color('#FF00FF')

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

            heatmap_yhat4=image_at_height(heatmap_path +'/'+str(i)+'_yhat_4.png')
            heatmap_yhat4.scale([0.29, 0.28, 0.28])
            heatmap_yhat4.move_to([0.96,0,0])
            heatmap_yhat4.set_opacity(0.5)
            heatmaps.add(heatmap_yhat4)

            canvas_x, canvas_y=latlong_to_canvas(xs[i][0], xs[i][1])
            training_point=m.Dot([canvas_x, canvas_y, 0], radius=0.012)
            if ys[i]==0.0: training_point.set_color('#00FFFF')
            elif ys[i]==1.0: training_point.set_color(YELLOW)
            elif ys[i]==2.0: training_point.set_color(GREEN)   
            elif ys[i]==3.0: training_point.set_color('#FF00FF')   

            step_label=m.Text("Step=")
            step_label.set_color(CHILL_BROWN)
            step_label.scale(0.12)
            step_label.move_to([1.3, -0.85, 0])

            step_count=m.Text(str(i).zfill(3))
            step_count.set_color(CHILL_BROWN)
            step_count.scale(0.12)
            step_count.move_to([1.43, -0.85, 0])

            self.add(step_label,step_count) 
            self.add(plane_1, plane_2, plane_3, plane_4)
            self.add(nums)
            # self.add(grad_regions) #I'm runnign out of steam here to do grad regions, leaving out for now - it's alraedy prettty complex!
            self.add(heatmaps)
            self.add(training_point)
            previous = [nums, plane_1, plane_2, plane_3, plane_4, heatmaps, training_point, step_label, step_count]
            self.wait(0.1)


        self.wait()
        pass




class p46_sketch(WelchScene):
    def construct(self):
        '''
        Ok starting with p45, I'll work on animating to shared p46 plot, and then start hacking on 3d. 
        '''
        data=np.load(data_path+'/cities_2d_1.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:15]
        grads=data[:,15:27]
        logits=data[:,27:31]
        yhats=data[:, 31:]


        net_background=m.SVGMobject(svg_path+'/p_48_background_1.svg')
        self.add(net_background)

        self.set_view(0, 0, 0, (-0.03, 0.01, 0.0), 2.01)
        europe_map=image_at_height(svg_path +'/map_cropped_one.png')
        europe_map.scale(0.28)
        europe_map.move_to([0.96,0,0])
        self.add(europe_map)


        i=300
        vertical_viz_scale=0.4
        nums=get_dem_numbers(i, xs, weights, logits, yhats)
        self.add(nums)

        axes_1 = diagram_axes_3d(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_1.move_to([-0.80, 0.7, 0])
        axes_1.rotate(-90*m.DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        axes_1.rotate(-30*m.DEGREES, [0,1,0]) #Twist around vertical
        plane_1=LinearPlane(axes_1, weights[i,0], weights[i,1], weights[i,8], vertical_viz_scale=vertical_viz_scale)
        plane_1.set_opacity(0.6)
        plane_1.set_color('#00FFFF')


        axes_2 = diagram_axes_3d(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_2.move_to([-0.80, 0.24, 0])
        axes_2.rotate(-90*m.DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        axes_2.rotate(-30*m.DEGREES, [0,1,0]) #Twist around vertical
        plane_2=LinearPlane(axes_2, weights[i,2], weights[i,3], weights[i,9], vertical_viz_scale=vertical_viz_scale)
        plane_2.set_opacity(0.6)
        plane_2.set_color(YELLOW)

        axes_3 = diagram_axes_3d(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_3.move_to([-0.80, -0.22, 0])
        axes_3.rotate(-90*m.DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        axes_3.rotate(-30*m.DEGREES, [0,1,0]) #Twist around vertical
        plane_3=LinearPlane(axes_3, weights[i,4], weights[i,5], weights[i,10], vertical_viz_scale=vertical_viz_scale)
        plane_3.set_opacity(0.6)
        plane_3.set_color(GREEN)

        axes_4 = diagram_axes_3d(
            x_range=[-15, 15, 1],
            y_range=[-15, 15, 1],
            z_range=[-10, 10, 1],
            width=0.28,
            height=0.28,
            depth=0.28,
            axis_config={
                "color": CHILL_BROWN,
                "include_ticks": False,
                "include_numbers": False,
                "include_tip": True,
                "stroke_width":2,
                "tip_config": {"width":0.015, "length":0.015}
                }
        )

        axes_4.move_to([-0.80, -0.7, 0])
        axes_4.rotate(-90*m.DEGREES, [1,0,0]) #Flip up #Going to need ot noodle with rotation to match map
        axes_4.rotate(-30*m.DEGREES, [0,1,0]) #Twist around vertical
        plane_4=LinearPlane(axes_4, weights[i,6], weights[i,7], weights[i,11], vertical_viz_scale=vertical_viz_scale)
        plane_4.set_opacity(0.6)
        plane_4.set_color('#FF00FF')


        self.add(axes_1, plane_1, axes_2, plane_2, axes_3, plane_3, axes_4, plane_4)

        self.wait()


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

        heatmap_yhat4=image_at_height(heatmap_path +'/'+str(i)+'_yhat_4.png')
        heatmap_yhat4.scale([0.29, 0.28, 0.28])
        heatmap_yhat4.move_to([0.96,0,0])
        heatmap_yhat4.set_opacity(0.5)
        heatmaps.add(heatmap_yhat4)

        canvas_x, canvas_y=latlong_to_canvas(xs[i][0], xs[i][1])
        training_point=m.Dot([canvas_x, canvas_y, 0], radius=0.012)
        if ys[i]==0.0: training_point.set_color('#00FFFF')
        elif ys[i]==1.0: training_point.set_color(YELLOW)
        elif ys[i]==2.0: training_point.set_color(GREEN)   
        elif ys[i]==3.0: training_point.set_color('#FF00FF')   

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
        # self.add(grad_regions) #I'm runnign out of steam here to do grad regions, leaving out for now - it's alraedy prettty complex!
        self.add(heatmaps)
        self.add(training_point)


        self.wait()
        pass

































