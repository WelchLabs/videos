from welch_paths import welch_path
import manimgx as m
from welch_geometry import diagram_axes_3d
import numpy as np
from welch_axes import WelchXAxis, WelchYAxis
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from welch_text import math_tex

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


class Dot3D(m.Sphere):
    def __init__(self, center=m.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)

data_dir=welch_path('backpropagation/hackin/')


class P31(WelchScene):
    def construct(self):

        surf=2.0*np.load(data_dir+'p_24_28_losses_4.npy')+0.8 #Adding a scaling factor here to make graph steeper, will need ot adjust tick labels
        xy=np.load(data_dir+'p_24_28_losses_4xy.npy')

        # Create the surface
        axes = diagram_axes_3d(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            z_range=[0.0, 3.5, 0.6],
            height=5,
            width=5,
            depth=4,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )

        
        # Add labels
        x_label = math_tex(r'\theta_{1}', font_size=40).set_color(CHILL_BROWN)
        y_label = math_tex(r'\theta_{2}', font_size=40).set_color(CHILL_BROWN)
        z_label = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, m.RIGHT)
        y_label.next_to(axes.y_axis, m.UP)
        z_label.next_to(axes.z_axis, m.OUT)
        z_label.rotate(90*m.DEGREES, [1,0,0])

        def param_surface(u, v):
            u_idx = np.abs(xy[0] - u).argmin()
            v_idx = np.abs(xy[1] - v).argmin()
            try:
                z = surf[u_idx, v_idx]
            except IndexError:
                z = 0
            return np.array([u, v, z])

        # Create main surface
        surface = parametric_surface(
            param_surface,
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(512, 512),
        )
        
        ts = textured_surface(surface, data_dir+'p_24_28_losses_4.png')
        set_surface_shading(ts, 0.0, 0.1, 0)
        ts.set_opacity(0.7)

        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 1024  # Number of points per line
        u_gridlines = m.VGroup()
        v_gridlines = m.VGroup()
        
        # Create u-direction gridlines
        u_values = np.linspace(-1, 4, num_lines)
        v_points = np.linspace(-1, 4, num_points)
        

        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = m.VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-1, 4, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface(u, v) for u in u_points]
            line = m.VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            v_gridlines.add(line)

        #i think there's a better way to do this
        offset=surface.get_corner((m.config.frame_height / 2 * m.DOWN)+m.LEFT)-axes.get_corner((m.config.frame_height / 2 * m.DOWN)+m.LEFT)
        axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);
        # axes.move_to([1,1,1])
        
        
        # Add everything to the scene
        self.add(axes, x_label, y_label, z_label)

        # self.add(u_gridlines, v_gridlines)
        # self.set_view(10, 57, 0, (2.09, 1.11, 1.36), 7.55)
        self.set_view(-24, 60, 0, (1.71, 1.77, 2.21), 8.03)
        self.wait()
        self.play(m.Create(u_gridlines), m.Create(v_gridlines),
            self.animate_view(42, 46, 0, (1.56, 1.37, 2.4), 7.78), run_time=4)
        self.play(m.FadeIn(ts))
        self.wait()

       
        #Bottom of Bowl
        u_min=0.95 
        v_min=2.3
        global_min=Dot3D(center=param_surface(u_min,v_min), radius=0.1, color='$FF00FF')
        self.add(global_min)
        self.wait()

        self.play(self.animate_view(0, 48, 0, (1.47, 1.55, 2.62), 7.78), run_time=6)





        # self.play(self.animate_view(-70, 49, 0, (1.59, 1.52, 2.39), 6.84), run_time=10, rate_func=linear)
        self.wait()




        pass
        self.wait(20)



class P24v4(WelchScene):
    def construct(self):

        surf=2.0*np.load(data_dir+'p_24_28_losses_4.npy')+0.8 #Adding a scaling factor here to make graph steeper, will need ot adjust tick labels
        xy=np.load(data_dir+'p_24_28_losses_4xy.npy')

        # Create the surface
        axes = diagram_axes_3d(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            z_range=[0.0, 3.5, 0.6],
            height=5,
            width=5,
            depth=4,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )


        
        # Add labels
        x_label = math_tex(r'\theta_{1}', font_size=40).set_color(CHILL_BROWN)
        y_label = math_tex(r'\theta_{2}', font_size=40).set_color(CHILL_BROWN)
        z_label = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, m.RIGHT)
        y_label.next_to(axes.y_axis, m.UP)
        z_label.next_to(axes.z_axis, m.OUT)
        z_label.rotate(90*m.DEGREES, [1,0,0])

        def param_surface(u, v):
            u_idx = np.abs(xy[0] - u).argmin()
            v_idx = np.abs(xy[1] - v).argmin()
            try:
                z = surf[u_idx, v_idx]
            except IndexError:
                z = 0
            return np.array([u, v, z])

        # Create main surface
        surface = parametric_surface(
            param_surface,
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(512, 512),
        )
        
        ts = textured_surface(surface, data_dir+'p_24_28_losses_4.png')
        set_surface_shading(ts, 0.0, 0.1, 0)
        ts.set_opacity(0.7)

        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 1024  # Number of points per line
        u_gridlines = m.VGroup()
        v_gridlines = m.VGroup()
        
        # Create u-direction gridlines
        u_values = np.linspace(-1, 4, num_lines)
        v_points = np.linspace(-1, 4, num_points)
        

        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = m.VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-1, 4, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface(u, v) for u in u_points]
            line = m.VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            v_gridlines.add(line)

        #i think there's a better way to do this
        offset=surface.get_corner((m.config.frame_height / 2 * m.DOWN)+m.LEFT)-axes.get_corner((m.config.frame_height / 2 * m.DOWN)+m.LEFT)
        axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);
        # axes.move_to([1,1,1])
        
        
        # Add everything to the scene
        self.add(axes, x_label, y_label, z_label)

        # self.add(u_gridlines, v_gridlines)
        # self.set_view(10, 57, 0, (2.09, 1.11, 1.36), 7.55)
        self.set_view(-24, 60, 0, (1.71, 1.77, 2.21), 8.03)
        self.wait()
        self.play(m.Create(u_gridlines), m.Create(v_gridlines), self.animate_view(38, 61, 0, (1.33, 1.43, 2.61), 7.78), run_time=4)
        self.play(m.FadeIn(ts))
        self.wait()

        #Ok now we want to move to and draw the curve for the first parameter projection. 
        self.play(self.animate_view(0, 91, 0, (1.71, 1.05, 3.47), 6.46),
                  x_label.animate.rotate(90*m.DEGREES, [1,0,0]),
                  ts.animate.set_opacity(0.2),
                  u_gridlines.animate.set_stroke(opacity=0.15),
                  v_gridlines.animate.set_stroke(opacity=0.15),
                  run_time=3)
        self.wait()

        u_points = np.linspace(-1, 4, num_points)
        points = [param_surface(u, 0) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_1 = m.VMobject()
        slice_1.set_points_smoothly(points)
        slice_1.set_stroke(width=4, color=YELLOW, opacity=0.8)

        self.play(m.Create(slice_1), run_time=2)
        self.wait()

        slice_t_bottom=m.Dot(param_surface(1.61, 0), radius=0.08, fill_color=YELLOW)
        slice_t_bottom.rotate(90*m.DEGREES, [1,0,0])
        self.add(slice_t_bottom)
        self.wait()

        y_label.rotate(90*m.DEGREES, [0,0,1]) #Maybe some way to hide this jump? Can't figure out how to do it in one step
        self.play(self.animate_view(79, 98, 0, (1.71, 1.59, 3.29), 6.21),
                  y_label.animate.rotate(90*m.DEGREES, [0,1,0]),
                  run_time=4)
        self.wait()
        

        v_points = np.linspace(-1, 4, num_points)
        points = [param_surface(1.61, v) for v in v_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_2 = m.VMobject()
        slice_2.set_points_smoothly(points)
        slice_2.set_stroke(width=4, color=BLUE, opacity=0.8)   

        self.play(m.Create(slice_2), run_time=2)
        self.wait()


        #note sure about camera movement here yet. Maybe camear moves while I draw the second curve? while making curve more opaque?
        u_points = np.linspace(-1, 4, num_points)
        points = [param_surface(u, 1.76) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_3 = m.VMobject()
        slice_3.set_points_smoothly(points)
        slice_3.set_stroke(width=4, color=YELLOW, opacity=0.8)
        self.wait()

        slice_bottom_2=m.Dot(param_surface(1.61, 1.76), radius=0.08, fill_color=BLUE)
        slice_bottom_2.rotate(90*m.DEGREES, [1,0,0])
        slice_bottom_2.rotate(90*m.DEGREES, [0,0,1])
        self.add(slice_bottom_2)
        self.wait()

        #Ok i think it makes sense to quickly draw then then move the camera. 
        self.play(m.Create(slice_3))

        self.wait()
        self.play(self.animate_view(1, 96, 0, (1.4, 1.57, 3.52), 6.57),
                  slice_1.animate.set_stroke(opacity=0.4), run_time=4)
        self.wait()

        slice_bottom_3=m.Dot(param_surface(1.11, 1.76), radius=0.08, fill_color=YELLOW)
        slice_bottom_3.rotate(90*m.DEGREES, [1,0,0])
        self.add(slice_bottom_3)
        self.wait()

        self.play(self.animate_view(42, 46, 0, (1.5, 1.67, 2.4), 7.61),
                 ts.animate.set_opacity(0.6),
                 u_gridlines.animate.set_stroke(opacity=0.2),
                 v_gridlines.animate.set_stroke(opacity=0.2), run_time=4)
        self.wait()

        self.play(
            slice_1.animate.set_opacity(0.0),
            slice_2.animate.set_opacity(0.0),
            slice_3.animate.set_opacity(0.0),
            slice_bottom_2.animate.set_opacity(0.0),
            slice_t_bottom.animate.set_opacity(0.0),
            slice_bottom_3.animate.set_opacity(0.0),
            self.animate_view(0, 49, 0, (1.48, 1.79, 2.54), 7.61),
            run_time=6
        )
        self.wait()


        #Bottom of Bowl
        u_min=0.95 
        v_min=2.3
        global_min=Dot3D(center=param_surface(u_min,v_min), radius=0.1, color='$FF00FF')
        self.add(global_min)
        self.wait()

        self.play(ts.animate.set_opacity(0.0), 
                  u_gridlines.animate.set_opacity(0.0),
                  v_gridlines.animate.set_opacity(0.0),
                  x_label.animate.rotate(-90*m.DEGREES, [1,0,0]),
                  y_label.animate.rotate(-90*m.DEGREES, [0,1,0]).rotate(-90*m.DEGREES, [0,0,1]),
                  self.animate_view(0, 0, 0, (1.54, 1.54, 2.52), 5.59), #overhead
                  run_time=6)





        # self.play(self.animate_view(-70, 49, 0, (1.59, 1.52, 2.39), 6.84), run_time=10, rate_func=linear)
        self.wait()




        pass
        self.wait(20)
        

class P24v5(WelchScene):
    def construct(self):

        surf=2.0*np.load(data_dir+'p_24_28_losses_4.npy')+0.8 #Adding a scaling factor here to make graph steeper, will need ot adjust tick labels
        xy=np.load(data_dir+'p_24_28_losses_4xy.npy')

        # Create the surface
        axes = diagram_axes_3d(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            z_range=[0.0, 3.5, 0.6],
            height=5,
            width=5,
            depth=4,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )
        
        # Add labels
        x_label = math_tex(r'\theta_{1}', font_size=40).set_color(CHILL_BROWN)
        y_label = math_tex(r'\theta_{2}', font_size=40).set_color(CHILL_BROWN)
        z_label = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, m.RIGHT)
        y_label.next_to(axes.y_axis, m.UP)
        z_label.next_to(axes.z_axis, m.OUT)
        z_label.rotate(90*m.DEGREES, [1,0,0])

        def param_surface(u, v):
            u_idx = np.abs(xy[0] - u).argmin()
            v_idx = np.abs(xy[1] - v).argmin()
            try:
                z = surf[u_idx, v_idx]
            except IndexError:
                z = 0
            return np.array([u, v, z])

        # Create main surface
        surface = parametric_surface(
            param_surface,
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(512, 512),
        )
        
        ts = textured_surface(surface, data_dir+'p_24_28_losses_4.png')
        set_surface_shading(ts, 0.0, 0.1, 0)
        ts.set_opacity(0.7)

        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 1024  # Number of points per line
        u_gridlines = m.VGroup()
        v_gridlines = m.VGroup()
        
        # Create u-direction gridlines
        u_values = np.linspace(-1, 4, num_lines)
        v_points = np.linspace(-1, 4, num_points)
        

        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = m.VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-1, 4, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface(u, v) for u in u_points]
            line = m.VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            v_gridlines.add(line)

        #i think there's a better way to do this
        offset=surface.get_corner((m.config.frame_height / 2 * m.DOWN)+m.LEFT)-axes.get_corner((m.config.frame_height / 2 * m.DOWN)+m.LEFT)
        axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);
        # axes.move_to([1,1,1])
        
        
        # Add everything to the scene
        self.add(axes, x_label, y_label, z_label)

        # self.add(u_gridlines, v_gridlines)
        # self.set_view(10, 57, 0, (2.09, 1.11, 1.36), 7.55)
        self.set_view(-24, 60, 0, (1.71, 1.77, 2.21), 8.03)
        self.wait()
        self.play(m.Create(u_gridlines), m.Create(v_gridlines), self.animate_view(38, 61, 0, (1.33, 1.43, 2.61), 7.78), run_time=4)
        self.play(m.FadeIn(ts))
        self.wait()

        #Ok I'm tempted to roll right into 29 now, it could be a nice transition I think
        # Can i get the grid lines to fall to the ground smoothly?

        def get_fallen_points(line, progress):
            # Get original points
            original_points = line.get_points()
            fallen_points = np.array(original_points)
            
            # Gradually reduce z-coordinate based on progress
            fallen_points[:, 2] = original_points[:, 2] * (1 - progress)
            
            return fallen_points

        # Create the falling animation for u_gridlines
        falling_animations_u = []
        for line in u_gridlines:
            falling_animations_u.append(
                m.UpdateFromAlphaFunc(
                    line,
                    lambda mob, alpha: mob.set_points(get_fallen_points(mob, alpha/6.0)) #Needs to match runtime
                )
            )

        # Create the falling animation for v_gridlines
        falling_animations_v = []
        for line in v_gridlines:
            falling_animations_v.append(
                m.UpdateFromAlphaFunc(
                    line,
                    lambda mob, alpha: mob.set_points(get_fallen_points(mob, alpha/6.0))
                )
            )
        self.wait()

        # Play both animations together
        self.play(
            *falling_animations_u,
            *falling_animations_v,
            m.FadeOut(axes),
            m.FadeOut(x_label),
            m.FadeOut(y_label),
            m.FadeOut(z_label),
            ts.animate.set_opacity(0.0),
            # slice_1.animate.set_opacity(0.0),
            # slice_2.animate.set_opacity(0.0),
            # slice_3.animate.set_opacity(0.0),
            #global_min.animate.set_opacity(0.00), 
            # slice_bottom_2.animate.set_opacity(0.0),
            # slice_t_bottom.animate.set_opacity(0.0),
            # slice_bottom_3.animate.set_opacity(0.0),
            u_gridlines.animate.set_stroke(opacity=0.6),
            v_gridlines.animate.set_stroke(opacity=0.6),
            self.animate_view(0, 0, 0, (1.62, 1.38, 2.23), 6.84),
            run_time=6
        )
        self.wait()

        intersection_dots = m.VGroup()
        for u in u_values:
            for v in u_values:  # Using same values since your grid is square
                # point = param_surface(u, v)
                dot = m.Dot([u, v, 0], radius=0.03, fill_color=m.WHITE, fill_opacity=0.8)
                intersection_dots.add(dot)
        self.play(m.Create(intersection_dots))
        self.wait()
        #Ok so i think from ehre we take over in illustrator/premier. 
        #I'll do the 3d version in a separate p29c manim file. 
        # self.play(ts.animate.set)

        pass
        self.wait(20)


class P24_2d(WelchScene):
    def construct(self):

        surf=np.load(data_dir+'p_24_28_losses_4.npy') #Adding a scaling factor here to make graph steeper, will need ot adjust tick labels
        xy=np.load(data_dir+'p_24_28_losses_4xy.npy')


        def param_surface(u, v):
            u_idx = np.abs(xy[0] - u).argmin()
            v_idx = np.abs(xy[1] - v).argmin()
            try:
                z = surf[u_idx, v_idx]
            except IndexError:
                z = 0
            return np.array([u, v, z])

        # Create main surface
        surface = parametric_surface(
            param_surface,
            u_range=[-1, 4],
            v_range=[-1, 4],
            resolution=(512, 512),
        )


        x_axis_1=WelchXAxis(x_min=-1.0, x_max=4.0, x_ticks=[-1.0, 0 ,1.0, 2.0, 3.0,], x_tick_height=0.15,        
                    x_label_font_size=22, stroke_width=3.0, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0.2, y_max=1.6, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4], y_tick_width=0.15,        
                  y_label_font_size=22, stroke_width=3.0, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_1 = math_tex(r'\theta_1', font_size=36).set_color(CHILL_BROWN)
        y_label_1 = math_tex('Loss', font_size=28).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, m.RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, m.UP, buff=0.08)

        num_points=1024
        u_points = np.linspace(-1, 4, num_points)
        points = [param_surface(u, 0) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        points=np.array(points)

        mapped_x_1=x_axis_1.map_to_canvas(points[:, 0]) 
        mapped_y_1=y_axis_1.map_to_canvas(points[:, 2])

        curve_1=m.VMobject()
        curve_1.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_1.set_stroke(width=4, color=YELLOW, opacity=1.0)

        # self.set_view(0, 89, 0, (-12.43, -0.01, 0.0), 4.99)
        self.set_view(0, 0, 0, (1.44, -0.6, 0.0), 8.36)
        self.add(x_axis_1, y_axis_1, x_label_1, y_label_1)
        self.wait()
        self.play(m.Create(curve_1), run_time=5)
        self.wait()




        x_axis_2=WelchXAxis(x_min=-1.0, x_max=4.0, x_ticks=[-1.0, 0 ,1.0, 2.0, 3.0,], x_tick_height=0.15,        
                    x_label_font_size=22, stroke_width=3.0, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_2=WelchYAxis(y_min=0.2, y_max=1.6, y_ticks=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4], y_tick_width=0.15,        
                  y_label_font_size=22, stroke_width=3.0, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_2 = math_tex(r'\theta_2', font_size=36).set_color(CHILL_BROWN)
        y_label_2 = math_tex('Loss', font_size=28).set_color(CHILL_BROWN)
        x_label_2.next_to(x_axis_2, m.RIGHT, buff=0.05)
        y_label_2.next_to(y_axis_2, m.UP, buff=0.08)

        num_points=1024
        v_points = np.linspace(-1, 4, num_points)
        points = [param_surface(1.61, v) for v in v_points]
        points=np.array(points)

        mapped_x_1=x_axis_2.map_to_canvas(points[:, 1]) 
        mapped_y_1=y_axis_2.map_to_canvas(points[:, 2])

        curve_2=m.VMobject()
        curve_2.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_2.set_stroke(width=4, color=BLUE, opacity=1.0)

        # self.set_view(0, 89, 0, (-12.43, -0.01, 0.0), 4.99)
        curve_2_group=m.VGroup(x_axis_2, y_axis_2, x_label_2, y_label_2, curve_2)
        curve_2_group.shift([0, -4, 0])

        self.add(x_axis_2, y_axis_2, x_label_2, y_label_2)
        self.wait()
        self.play(m.Create(curve_2), run_time=5)
        self.wait()


        num_points=1024
        u_points = np.linspace(-1, 4, num_points)
        points = [param_surface(u, 1.76) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        points=np.array(points)

        mapped_x_1=x_axis_1.map_to_canvas(points[:, 0]) 
        mapped_y_1=y_axis_1.map_to_canvas(points[:, 2])

        curve_3=m.VMobject()
        curve_3.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_3.set_stroke(width=4, color=YELLOW, opacity=1.0)

        self.play(m.Create(curve_3),
                 curve_1.animate.set_stroke(color='#806a2d'), #'#7F5816'), #Darker yellow, opacity wasn't working
                run_time=5)

        self.wait(20)


        # ffd35a

        # slice_1 = VMobject()
        # slice_1.set_points_smoothly(points)
        # slice_1.set_stroke(width=4, color=YELLOW, opacity=0.8)

        # v_points = np.linspace(-1, 4, num_points)
        # points = [param_surface(1.61, v) for v in v_points] #Theta2 isn't exactly 0, but pretty close. 
        # slice_2 = VMobject()
        # slice_2.set_points_smoothly(points)
        # slice_2.set_stroke(width=4, color=BLUE, opacity=0.8)  

        # u_points = np.linspace(-1, 4, num_points)
        # points = [param_surface(u, 1.76) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        # slice_3 = VMobject()
        # slice_3.set_points_smoothly(points)
        # slice_3.set_stroke(width=4, color=YELLOW, opacity=0.8) 


        # curve_1=VMobject()         
        # curve_1.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        # curve_1.set_stroke(width=4, color=YELLOW, opacity=1.0)

        # axes_1=VGroup(x_axis_1, y_axis_1, x_label_1, y_label_1, curve_1)
        # axes_1.move_to([-12.5, 0, 0]) #Start way over here, so the final 3d column of plots land close to origin
        # axes_1.rotate(90*DEGREES, [1,0,0], about_point=ORIGIN)
        # self.set_view(0, 89, 0, (-12.43, -0.01, 0.0), 4.99)
        # self.add(x_axis_1, y_axis_1, x_label_1, y_label_1)
        # self.wait(0)
        # self.play(ShowCreation(curve_1), run_time=5)
