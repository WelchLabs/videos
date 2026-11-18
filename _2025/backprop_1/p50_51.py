from welch_paths import welch_path
import manimgx as m
from welch_scene import manual_camera_interpolation
from landscape_geometry import get_numerical_gradient
from welch_geometry import diagram_axes_3d
import numpy as np
from welch_axes import WelchXAxis, WelchYAxis
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from welch_text import math_tex

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

save_dir=welch_path('backpropagation/hackin/p50/')
xy=np.linspace(-2.5, 2.5, 256)


class Dot3D(m.Sphere):
    def __init__(self, center=m.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)



class P50a(WelchScene):
    def construct(self):
        '''
        Ok, this has been like a gruelling 10 day animation process - last big scene here. 
        I need a 2d and 3d view of the same thing -> I don't think there's a need to do 2d/3d transitions at this point
        like it might be nice - but I'm already running behind and I think I can make it pretty clear 
        from 2 differen scenes. 

        So I'll do a 2d and 3d version of the same scene. 
        I think let's do the 3d frist, then the 2d can be little slices of that.  

        Yo claude! I'm trying to make a demonstration "loss landscape" that has some special properties, I'm using python. Can you help me create a 2d numpy array that contains 2 parabaoild shapes? One large/gradual one that takes up the whole space (let's do x=-2.5->2.5 and y=-2.5->2.5), and one small parabaloid that's like a "little local valley"? I don't want the small one in the center.
        '''


        surf=np.load(save_dir+'p50_2d.npy') #Optionally add scaling factor here

        # Create the surface
        axes = diagram_axes_3d(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            z_range=[0.0, 3.5, 1.0],
            height=5,
            width=5,
            depth=3.5,
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
            u_idx = np.abs(xy - u).argmin()
            v_idx = np.abs(xy - v).argmin()
            try:
                z = surf[u_idx, v_idx]
            except IndexError:
                z = 0
            return np.array([u, v, z])

        # Create main surface
        surface = parametric_surface(
            param_surface,
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(256, 256),
        )
        
        ts = textured_surface(surface, save_dir+'p50_2d.png')
        set_surface_shading(ts, 0.0, 0.1, 0)
        ts.set_opacity(0.7)

        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 256  # Number of points per line
        u_gridlines = m.VGroup()
        v_gridlines = m.VGroup()
        
        # Create u-direction gridlines
        u_values = np.linspace(-2.5, 2.5, num_lines)
        v_points = np.linspace(-2.5, 2.5, num_points)
        
        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = m.VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-2.5, 2.5, num_points)
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
        
        # Add everything to the scene
        self.set_view(23, 53, 0, (0.11, 0.34, 0.31), 8.28)
        self.add(axes[:2], x_label, y_label)#, z_label) #Don't really need vertical axis here!
        self.wait()

        self.play(m.Create(ts),
                m.Create(u_gridlines),
                 m.Create(v_gridlines),
                 run_time=6.0)
        self.wait()

        self.play(self.animate_view(31, 44, 0, (-0.03, 0.45, 0.53), 7.36), run_time=4.0)
        self.wait()

        # Alrighty, now I need to visualize slices and run gradient descent
        # I could do analytical, but learning towards just doing numerical with velocity - that worked fine before
        
        slice_1_index=-0.5
        u_points = np.linspace(-2.5, 2.5, num_points)
        points = [param_surface(u, slice_1_index) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_1 = m.VMobject()
        slice_1.set_points_smoothly(points)
        slice_1.set_stroke(width=4, color=YELLOW, opacity=1.0)
        self.wait()

        self.play(m.Create(slice_1))
        self.wait()

        starting_coords=[-1,-0.5] 
        starting_point=param_surface(*starting_coords)
        s1=Dot3D(center=starting_point, radius=0.1, color='$FF00FF')
        self.add(s1)


        num_steps=128 # I think it gest stuck around 30-40 at lr -0.01 - play with this for smoother/longer animation
        learning_rate=3e-3
        momentum = 0.95 #lil momentum to get to a decent minima lol
        trajectory=[[starting_point[0], starting_point[1], param_surface(starting_point[0], starting_point[1])[2]]]
        velocity = np.zeros(2)
        for i in range(num_steps):
            g=get_numerical_gradient(param_surface, trajectory[-1][0], trajectory[-1][1], epsilon=0.01)
            velocity = momentum * velocity - learning_rate * np.array(g)
            new_x = trajectory[-1][0] + velocity[0]
            new_y = trajectory[-1][1] + velocity[1]
            trajectory.append([new_x, new_y, param_surface(new_x, new_y)[2]])

        #I don't think we need/want a camera move here. 
        #I think add slices every N steps - maybe with little bit different opacities?

        way_point_1=48
        way_point_2=71
        way_point_3=96

        t = m.VMobject()
        t.set_stroke(width=5, color="#FF00FF", opacity=0.9)
        self.add(t)
        for i in range(way_point_1): #Go partially and then add countour
            s1.move_to(trajectory[i])
            t.set_points_smoothly(trajectory[:i])
            self.wait(0.1)
        self.wait()
        self.wait()

        slice_2_index=trajectory[way_point_1][1]
        u_points = np.linspace(-2.5, 2.5, num_points)
        points = [param_surface(u, slice_2_index) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_2 = m.VMobject()
        slice_2.set_points_smoothly(points)
        slice_2.set_stroke(width=4, color=YELLOW, opacity=0.7)
        self.wait()

        self.play(m.Create(slice_2))
        self.wait()

        for i in range(way_point_1, way_point_2): #Go partially and then add countour
            s1.move_to(trajectory[i])
            t.set_points_smoothly(trajectory[:i])
            self.wait(0.1)
        self.wait()

        slice_3_index=trajectory[way_point_2][1]
        u_points = np.linspace(-2.5, 2.5, num_points)
        points = [param_surface(u, slice_3_index) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_3 = m.VMobject()
        slice_3.set_points_smoothly(points)
        slice_3.set_stroke(width=4, color=YELLOW, opacity=0.5)
        self.wait()

        self.play(m.Create(slice_3))
        self.wait()


        start_orientation=[31, 44, 0, (-0.03, 0.45, 0.53), 7.36]
        end_orientation=[38, 37, 0, (-0.0, 0.47, 0.53), 7.94]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=way_point_3-way_point_2)
        self.wait()

        for i in range(way_point_2, way_point_3): #Go partially and then add countour
            s1.move_to(trajectory[i])
            t.set_points_smoothly(trajectory[:i])
            self.set_view(*interp_orientations[i-way_point_2])
            self.wait(0.1)
        self.wait()


        slice_4_index=trajectory[way_point_3][1]
        u_points = np.linspace(-2.5, 2.5, num_points)
        points = [param_surface(u, slice_4_index) for u in u_points] #Theta2 isn't exactly 0, but pretty close. 
        slice_4 = m.VMobject()
        slice_4.set_points_smoothly(points)
        slice_4.set_stroke(width=4, color=YELLOW, opacity=0.3)
        self.wait()

        self.play(m.Create(slice_4))
        self.wait()


        # self.remove(ts)


        ## Hmm why is this fucker changing color?
        # dot_path=Group()
        # self.add(dot_path)
        # for i in range(num_steps): #First leg -> getting stuck
        #     s1.move_to(trajectory[i]) #Get above surface to not fuck up opacity
        #     dot_path.add(Dot3D(center=trajectory[i], radius=0.03, color='$FF00FF').set_opacity(0.8))
        #     # self.wait(0.1)
        # self.wait()

        # dot_path.set_opacity(0.5)
        # self.add(dot_path)
        # self.remove(dot_path)

        # Is it crazy to clear everything and to the 2d animation in the same sequence? 
        # Let's see here...

        self.remove(slice_1, slice_2, slice_3, slice_4, ts, u_gridlines, v_gridlines, axes, s1, t, x_label, y_label)
        # self.set_view(0, 0, 0, (0,0,0), 8.00)
        self.set_view(0, 0, 0, (5.16, 1.19, -0.0), 7.14)
        self.wait()

        x_axis_1=WelchXAxis(x_min=-2.5, x_max=2.5, x_ticks=[], x_tick_height=0.15,        
                            x_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=-1, y_max=2, y_ticks=[], y_tick_width=0.15,        
                          y_label_font_size=20, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=3)

        x_label_1 = math_tex(r'\theta_1', font_size=28).set_color(CHILL_BROWN)
        y_label_1 = math_tex('Loss', font_size=22).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, m.RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, m.UP, buff=0.08)


        axes_1=m.VGroup(x_axis_1, y_axis_1, x_label_1, y_label_1)
        self.add(axes_1)
        self.wait()

        points_1 = [param_surface(u, slice_1_index)[2] for u in u_points]
        mapped_x_1=x_axis_1.map_to_canvas(u_points) 
        mapped_y_1=y_axis_1.map_to_canvas(np.array(points_1))
        curve_1=m.VMobject()
        curve_1.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_1.set_stroke(width=4, color=YELLOW, opacity=1.0)
        self.play(m.Create(curve_1))
        self.wait()


        points_2 = [param_surface(u, slice_2_index)[2] for u in u_points]
        mapped_x_1=x_axis_1.map_to_canvas(u_points) 
        mapped_y_1=y_axis_1.map_to_canvas(np.array(points_2))
        curve_2=m.VMobject()
        curve_2.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_2.set_stroke(width=4, color=YELLOW, opacity=0.7)
        self.play(m.Create(curve_2))
        self.wait()

        points_2 = [param_surface(u, slice_3_index)[2] for u in u_points]
        mapped_x_1=x_axis_1.map_to_canvas(u_points) 
        mapped_y_1=y_axis_1.map_to_canvas(np.array(points_2))
        curve_2=m.VMobject()
        curve_2.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_2.set_stroke(width=4, color=YELLOW, opacity=0.5)
        self.play(m.Create(curve_2))
        self.wait()

        points_2 = [param_surface(u, slice_4_index)[2] for u in u_points]
        mapped_x_1=x_axis_1.map_to_canvas(u_points) 
        mapped_y_1=y_axis_1.map_to_canvas(np.array(points_2))
        curve_2=m.VMobject()
        curve_2.set_points_smoothly(np.vstack((mapped_x_1, mapped_y_1, np.zeros_like(mapped_x_1))).T)
        curve_2.set_stroke(width=4, color=YELLOW, opacity=0.3)
        self.play(m.Create(curve_2))
        self.wait()



        self.wait(20)
        pass












