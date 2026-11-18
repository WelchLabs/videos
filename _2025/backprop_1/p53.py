from welch_paths import welch_path
import manimgx as m
from welch_scene import manual_camera_interpolation
from welch_geometry import diagram_axes_3d
import numpy as np
from welch_axes import WelchXAxis, WelchYAxis
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from welch_text import math_tex
import matplotlib.pyplot as plt
from tqdm import tqdm
save_dir=welch_path('backpropagation/hackin/')


CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

class Dot3D(m.Sphere):
    def __init__(self, center=m.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)


# Parameters
num_points = 20
true_slope = 0.61
true_intercept = 1
noise_level = 2.2
learning_rate = 0.01
num_iterations = 1000 #1000

# Generate synthetic data
np.random.seed(2)  # For reproducibility
x_values = np.random.uniform(0, 8, num_points)
y_values = true_slope * x_values + true_intercept + (np.random.random(num_points) - 0.5) * noise_level

# Initialize model parameters
slope = 0.5
intercept = 2.0

predictions = slope * x_values + intercept
errors = predictions - y_values
loss = np.mean(errors ** 2)

slopes=[slope]
intercepts=[intercept]
losses=[loss]

for iteration in range(num_iterations):
    # Calculate gradients
    slope_gradient = 2 * np.mean(errors * x_values)
    intercept_gradient = 2 * np.mean(errors)
    
    # Update parameters
    new_slope = slope - learning_rate * slope_gradient
    new_intercept = intercept - learning_rate * intercept_gradient
    
    # Calculate new loss
    new_predictions = new_slope * x_values + new_intercept
    new_errors = new_predictions - y_values
    new_loss = np.mean(new_errors ** 2)
    
    # Update variables for next iteration
    slope = new_slope
    intercept = new_intercept
    errors = new_errors
    loss = new_loss

    slopes.append(slope)
    intercepts.append(intercept)
    losses.append(loss)

# Create a grid of x and y values - slopes and y-intercepts
slope_min=0.0
slope_max=1.0
y_int_min=0.0
y_int_max=2.0

landscape_slopes = np.linspace(slope_min, slope_max, 256) #Slopers
landscape_intercepts = np.linspace(y_int_min, y_int_max, 256) #Y-interecepts

z=[]
for s in tqdm(landscape_slopes):
    z.append([])
    for yi in landscape_intercepts:
        yhat = s * x_values + yi
        e = yhat - y_values
        l = np.mean(e ** 2)
        z[-1].append(l)
Z=np.array(z)

plt.figure(frameon=False)
ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
ax.set_axis_off()
plt.gcf().add_axes(ax)
plt.imshow(np.rot90(Z)) #have to transpose if transposing u and v and param_surface_1
plt.savefig(save_dir+'p53_2d.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()


class P53_3D_v3(WelchScene):
    def construct(self):
        surf = 3.5 * Z / Z.max()

        # Create the axes
        axes = diagram_axes_3d(
            x_range=[slope_min, slope_max, 1],
            y_range=[y_int_min, y_int_max, 2],
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
        x_label = math_tex(r'slope', font_size=40).set_color(CHILL_BROWN)
        y_label = math_tex(r'y-intercept', font_size=40).set_color(CHILL_BROWN)
        z_label = math_tex('Loss', font_size=30).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, m.RIGHT)
        y_label.next_to(axes.y_axis, m.UP)
        z_label.next_to(axes.z_axis, m.OUT)
        z_label.rotate(90*m.DEGREES, [1,0,0])

        # Define param_surface using axes.c2p to map coordinates correctly
        def param_surface(u, v):
            u_idx = np.abs(landscape_slopes - u).argmin()
            v_idx = np.abs(landscape_intercepts - v).argmin()
            try:
                z_val = surf[u_idx, v_idx]
            except IndexError:
                z_val = 0
            # Use axes.c2p to map from math coordinates to scene coordinates
            return axes.c2p(u, v, z_val)

        # Create main surface
        surface = parametric_surface(
            param_surface,
            u_range=[slope_min, slope_max],
            v_range=[y_int_min, y_int_max],
            resolution=(256, 256),
        )
        
        ts = textured_surface(surface, save_dir+'p53_2d.png')
        set_surface_shading(ts, 0.0, 0.1, 0)
        ts.set_opacity(0.7)

        # Create gridlines using polylines and axes.c2p
        num_lines = 20  # Number of gridlines in each direction
        num_points = 256  # Number of points per line
        u_gridlines = m.VGroup()
        v_gridlines = m.VGroup()
        
        # Create u-direction gridlines using axes.c2p
        u_values = np.linspace(slope_min, slope_max, num_lines)
        v_points = np.linspace(y_int_min, y_int_max, num_points)
        
        for u in u_values:
            points = [axes.c2p(u, v, param_surface(u, v)[2]) for v in v_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines using axes.c2p
        v_values = np.linspace(y_int_min, y_int_max, num_lines)
        u_points = np.linspace(slope_min, slope_max, num_points)
        for v in v_values:
            points = [axes.c2p(u, v, param_surface(u, v)[2]) for u in u_points]
            line = m.VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(width=1, color=m.WHITE, opacity=0.3)
            v_gridlines.add(line)

        # Group surface and gridlines
        surface_group = m.Group(ts, u_gridlines, v_gridlines)
        
        # Add the elements to the scene
        self.add(axes[:2], x_label, y_label) #, z_label)
        self.add(surface_group)

        # Set camera angle
        self.set_view(0, 27, 0, (0.85, 1.29, 0.26), 9.73)
        self.wait()


        # reorient(0, 8, 0, (0.06, -0.01, 0.09), 9.62)

        # Create and add trajectory
        t = m.VMobject()
        t.set_stroke(width=5, color="#FF00FF", opacity=0.9)
        s1=Dot3D(center=axes.c2p(slopes[0], intercepts[0],  3.5 * losses[0] / Z.max()), radius=0.09, color='$FF00FF')
        self.add(t)
        self.add(s1)

        # # Convert trajectory coordinates using axes.c2p
        # trajectory_points = []
        # for s, i, l in zip(slopes, intercepts, losses):
        #     # Map the point to the scene coordinates
        #     point = axes.c2p(s, i, 3.5 * l / Z.max())
        #     trajectory_points.append(point)

        # t.set_points_smoothly(trajectory_points)
        
        start_orientation=[0, 8, 0, (0.06, -0.01, 0.09), 9.62]
        end_orientation=[0, 18, 0, (0.06, -0.01, 0.09), 9.62]
        interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_iterations)

        for iter_count in range(num_iterations):
            self.remove(t)

            t = m.VMobject()
            t.set_stroke(width=5, color="#FF00FF", opacity=0.9)
            trajectory_points = []
            for s, i, l in zip(slopes[:iter_count], intercepts[:iter_count], losses[:iter_count]):
                # Map the point to the scene coordinates
                point = axes.c2p(s, i, 3.5 * l / Z.max())
                trajectory_points.append(point)
            t.set_points_smoothly(trajectory_points)
            self.add(t)

            s1.move_to(axes.c2p(slopes[iter_count], intercepts[iter_count],  3.5 * losses[iter_count] / Z.max()))
            self.set_view(*interp_orientations[iter_count])
            # self.wait(0.1)
            self.wait(1/30.)

        self.wait()

        self.play(m.FadeOut(axes[:2]), m.FadeOut(x_label), m.FadeOut(y_label), run_time=3)

        self.wait(20)
        pass


class P53_2D(WelchScene):
    def construct(self):

        x_axis_1=WelchXAxis(x_min=0, x_max=8.5, x_ticks=[2, 4, 6, 8], x_tick_height=0.15,        
                            x_label_font_size=22, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)
        y_axis_1=WelchYAxis(y_min=0, y_max=8.5, y_ticks=[2, 4, 6, 8], y_tick_width=0.15,        
                          y_label_font_size=22, stroke_width=2.5, arrow_tip_scale=0.1, axis_length_on_canvas=4)

        x_label_1 = math_tex('x', font_size=28).set_color(CHILL_BROWN)
        y_label_1 = math_tex('y', font_size=28).set_color(CHILL_BROWN)
        x_label_1.next_to(x_axis_1, m.RIGHT, buff=0.05)
        y_label_1.next_to(y_axis_1, m.UP, buff=0.08)


        axes_1=m.VGroup(x_axis_1, y_axis_1, x_label_1, y_label_1)
        self.add(axes_1)
        self.wait()

        mapped_x_1=x_axis_1.map_to_canvas(x_values) 
        mapped_y_1=y_axis_1.map_to_canvas(y_values)

        dots = m.VGroup()
        for i in range(num_points):
            dot = m.Dot([mapped_x_1[i], mapped_y_1[i],0], radius=0.06)
            dot.set_color(YELLOW)
            dot.set_opacity(0.95)
            dots.add(dot)

        self.add(dots)

        self.set_view(0, 0, 0, (-1.29, 1.85, 0.0), 8.00)

        line_points=np.array([[0, intercepts[0], 0],
                  [8, slopes[0]*8+intercepts[0], 0]])
        line_points_mapped=np.zeros_like(line_points)
        line_points_mapped[:,0]=x_axis_1.map_to_canvas(line_points[:,0]) 
        line_points_mapped[:,1]=y_axis_1.map_to_canvas(line_points[:,1])

        line = m.VGroup()
        line.set_points_smoothly(line_points_mapped)
        line.set_stroke(width=4, color=YELLOW, opacity=1.0)
        self.add(line)

        # axes.plot(lambda x: slope * x + intercept, color=RED)
        line_label = math_tex(f"y = {slope:.2f}x + {intercept:.2f}", font_size=24)
        line_label.next_to(axes_1, m.UP).shift(0.3*m.DOWN) #.shift(RIGHT * 1)
        line_label.set_color(YELLOW)
        self.add(line_label)


        #Ok so yeah I think we're good to go ahead and animate the 2d panel right?
        #Then will do 3d panel in a separate class

        for i in range(num_iterations):
            self.remove(line, line_label)
            line_points=np.array([[0, intercepts[i], 0],
                      [8, slopes[i]*8+intercepts[i], 0]])
            line_points_mapped=np.zeros_like(line_points)
            line_points_mapped[:,0]=x_axis_1.map_to_canvas(line_points[:,0]) 
            line_points_mapped[:,1]=y_axis_1.map_to_canvas(line_points[:,1])

            line = m.VGroup()
            line.set_points_smoothly(line_points_mapped)
            line.set_stroke(width=4, color=YELLOW, opacity=1.0)
            self.add(line)

            # axes.plot(lambda x: slope * x + intercept, color=RED)
            line_label = math_tex(f"y = {slopes[i]:.2f}x + {intercepts[i]:.2f}", font_size=24)
            line_label.next_to(axes_1, m.UP).shift(0.3*m.DOWN) #.shift(RIGHT * 1)
            line_label.set_color(YELLOW)
            self.add(line_label)
            self.wait(1/30.)


        self.wait()
        pass

# class P53_3D(InteractiveScene):
#     def construct(self):

#         surf=3.5*Z/Z.max()

#         # Create the surface
#         axes = ThreeDAxes(
#             x_range=[-1.5, 2.0, 1],
#             y_range=[-4, 5, 2],
#             z_range=[0.0, 3.5, 1.0],
#             height=5,
#             width=5,
#             depth=3.5,
#             axis_config={
#                 "include_ticks": True,
#                 "color": CHILL_BROWN,
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
#             }
#         )

        
#         # Add labels
#         x_label = Tex(r'slope', font_size=40).set_color(CHILL_BROWN)
#         y_label = Tex(r'y-intercept', font_size=40).set_color(CHILL_BROWN)
#         z_label = Tex('Loss', font_size=30).set_color(CHILL_BROWN)
#         x_label.next_to(axes.x_axis, RIGHT)
#         y_label.next_to(axes.y_axis, UP)
#         z_label.next_to(axes.z_axis, OUT)
#         z_label.rotate(90*DEGREES, [1,0,0])

#         def param_surface(u, v):
#             u_idx = np.abs(landscape_slopes - u).argmin()
#             v_idx = np.abs(landscape_intercepts - v).argmin()
#             try:
#                 z = surf[u_idx, v_idx]
#             except IndexError:
#                 z = 0
#             return np.array([u, v, z])

#         # Create main surface
#         surface = ParametricSurface(
#             param_surface,
#             u_range=[-1.5, 2.5],
#             v_range=[-4, 6],
#             resolution=(256, 256),
#         )
        
#         ts = TexturedSurface(surface, save_dir+'p53_2d.png')
#         ts.set_shading(0.0, 0.1, 0)
#         ts.set_opacity(0.7)

#         # Create gridlines using polylines instead of parametric curves
#         num_lines = 20  # Number of gridlines in each direction
#         num_points = 256  # Number of points per line
#         u_gridlines = VGroup()
#         v_gridlines = VGroup()
        
#         # Create u-direction gridlines
#         u_values = np.linspace(-1.5, 2.5, num_lines)
#         v_points = np.linspace(-4, 6, num_points)
        
#         for u in u_values:
#             points = [param_surface(u, v) for v in v_points]
#             line = VMobject()
#             # line.set_points_as_corners(points)
#             line.set_points_smoothly(points)
#             line.set_stroke(width=1, color=WHITE, opacity=0.3)
#             u_gridlines.add(line)
        
#         # Create v-direction gridlines
#         u_points = np.linspace(-1.5, 2.5, num_lines)
#         for v in np.linspace(-4, 6, num_lines):  # Using same number of lines for both directions
#             points = [param_surface(u, v) for u in u_points]
#             line = VMobject()
#             # line.set_points_as_corners(points)
#             line.set_points_smoothly(points)
#             line.set_stroke(width=1, color=WHITE, opacity=0.3)
#             v_gridlines.add(line)

#         #i think there's a better way to do this
#         groupy_group=Group(ts, u_gridlines, v_gridlines)
#         groupy_group.scale([1, 5.0/12, 1])
        

#         offset=ts.get_corner(BOTTOM+LEFT)-axes.get_corner(BOTTOM+LEFT)
#         axes.shift(offset); x_label.shift(offset); y_label.shift(offset); z_label.shift(offset);

#         # self.add(axes[:2], x_label, y_label)
#         groupy_group.shift([0, 0, 0.25])
#         self.add(axes[:2], x_label, y_label) # , z_label)
#         self.add(ts, u_gridlines, v_gridlines)

#         self.set_view(0, 27, 0, (0.85, 1.29, 0.26), 9.73) #Maybe a little camera move down while learning?
#         self.wait()

#         t = VMobject()
#         t.set_stroke(width=5, color="#FF00FF", opacity=0.9)
        

#         trajectory=np.vstack((np.array(slopes), np.array(intercepts), 3.5*np.array(losses)/Z.max())).T

#         t.set_points_smoothly(trajectory)
#         t.scale([1, 5.0/12, 1])

#         self.add(t)

#         ## Grrr getting stuck here -> need to revisit scaling when I come back. 

#         # for i in range(way_point_1): #Go partially and then add countour
#         #     s1.move_to(trajectory[i])
#         #     t.set_points_smoothly(trajectory[:i])
#         #     self.wait(0.1)
#         # self.wait()



#         self.wait()
#         self.embed()











# # Create the surface
# axes = ThreeDAxes(
#     x_range=[-1, 4, 1],
#     y_range=[-1, 4, 1],
#     z_range=[0.0, 3.5, 1.0],
#     height=5,
#     width=5,
#     depth=3.5,
#     axis_config={
#         "include_ticks": True,
#         "color": CHILL_BROWN,
#         "stroke_width": 2,
#         "include_tip": True,
#         "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
#     }
# )

# self.add(axes)
# self.embed()

# self.wait()




# Claude's first pass - might make sense to break down scene by scene. 
# Start with just the line fitting to data part panel. 

# from manimlib import *
# import numpy as np

# # Define colors
# CHILL_BROWN = '#948979'
# YELLOW = '#ffd35a'
# BLUE = '#65c8d0'
# ORANGE = '#ff7c4d'
# GREEN = '#7cda5f'

# class GradientDescentOnLinearModel(Scene):
#     def construct(self):
#         # Create synthetic data
#         np.random.seed(42)  # For reproducibility
#         num_points = 20
#         x_data = np.random.uniform(0, 10, num_points)
#         # True parameters: slope=2, intercept=1
#         true_theta1, true_theta2 = 2.0, 1.0
#         y_data = true_theta1 * x_data + true_theta2 + np.random.normal(0, 1, num_points)
        
#         # Create layout with two panels
#         self.create_split_screen(x_data, y_data, true_theta1, true_theta2)
        
#     def create_split_screen(self, x_data, y_data, true_theta1, true_theta2):
#         # Create left panel - 3D loss landscape
#         loss_surface_view = self.create_loss_surface_view(x_data, y_data)
#         loss_surface_view.scale(0.5).to_edge(LEFT)
        
#         # Create right panel - Data fitting view
#         data_fitting_view = self.create_data_fitting_view(x_data, y_data)
#         data_fitting_view.scale(0.5).to_edge(RIGHT)
        
#         # Add both panels to the scene
#         self.add(loss_surface_view, data_fitting_view)
        
#         # Run gradient descent
#         self.animate_gradient_descent(loss_surface_view, data_fitting_view, x_data, y_data)
        
#     def create_loss_surface_view(self, x_data, y_data):
#         # Create a group to hold all 3D objects
#         surface_group = VGroup()
        
#         # Create 3D axes
#         axes = ThreeDAxes(
#             x_range=[-1, 4, 1],
#             y_range=[-1, 4, 1],
#             z_range=[0.0, 10.0, 2.0],
#             height=6,
#             width=6,
#             depth=4,
#             axis_config={
#                 "include_ticks": True,
#                 "color": CHILL_BROWN,
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
#             }
#         )
        
#         # Add axis labels
#         x_label = Text("θ₁", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
#         y_label = Text("θ₂", font_size=24).next_to(axes.y_axis.get_end(), UP)
#         z_label = Text("Loss", font_size=24).next_to(axes.z_axis.get_end(), OUT)
        
#         axes_labels = VGroup(x_label, y_label, z_label)
        
#         # Create the loss surface
#         surface = self.create_loss_surface(axes, x_data, y_data)
        
#         # Add a title
#         title = Text("Loss Surface", font_size=30).to_edge(UP)
        
#         # Create a parameter point that will move on the surface
#         initial_theta1, initial_theta2 = 0.0, 0.0
#         initial_loss = self.loss_function(initial_theta1, initial_theta2, x_data, y_data)
#         parameter_point = Sphere(radius=0.1, color=ORANGE)
#         parameter_point.move_to(axes.c2p(initial_theta1, initial_theta2, initial_loss))
        
#         # Add elements to the group
#         surface_group.add(axes, axes_labels, surface, parameter_point, title)
        
#         # Set initial camera orientation for better view
#         frame = self.camera.frame
#         frame.set_euler_angles(phi=70 * DEGREES, theta=30 * DEGREES)
        
#         return surface_group
        
#     def create_loss_surface(self, axes, x_data, y_data):
#         # Function to compute z for any (x, y)
#         def param_surface(u, v):
#             theta1 = u
#             theta2 = v
#             loss = self.loss_function(theta1, theta2, x_data, y_data)
#             return axes.c2p(u, v, loss)
        
#         # Create the surface
#         surface = ParametricSurface(
#             param_surface,
#             u_range=[-1, 4],
#             v_range=[-1, 4],
#             resolution=(30, 30),
#             checkerboard_colors=[BLUE, BLUE.lighter(0.5)],
#             fill_opacity=0.7
#         )
        
#         return surface
    
#     def create_data_fitting_view(self, x_data, y_data):
#         # Create a group for data fitting visualization
#         fitting_group = VGroup()
        
#         # Create 2D axes
#         axes = Axes(
#             x_range=[0, 12, 2],
#             y_range=[-1, 25, 5],
#             height=6,
#             width=6,
#             axis_config={"include_tip": True, "color": CHILL_BROWN}
#         )
        
#         # Add axis labels
#         x_label = Text("x", font_size=24).next_to(axes.x_axis.get_end(), RIGHT)
#         y_label = Text("y", font_size=24).next_to(axes.y_axis.get_end(), UP)
#         axes_labels = VGroup(x_label, y_label)
        
#         # Plot data points
#         data_points = VGroup()
#         for i in range(len(x_data)):
#             point = Dot(axes.c2p(x_data[i], y_data[i]), color=YELLOW, radius=0.05)
#             data_points.add(point)
        
#         # Initial line with random parameters
#         initial_theta1, initial_theta2 = 0.0, 0.0
#         line_function = lambda x: initial_theta1 * x + initial_theta2
#         linear_model = axes.plot(line_function, x_range=[0, 12], color=GREEN)
        
#         # Add title
#         title = Text("Data Fitting", font_size=30).to_edge(UP)
        
#         # Add all elements to the group
#         fitting_group.add(axes, axes_labels, data_points, linear_model, title)
        
#         return fitting_group
    
#     def animate_gradient_descent(self, loss_view, data_view, x_data, y_data):
#         # Initialize parameters
#         theta1, theta2 = 0.0, 0.0
#         learning_rate = 0.01
#         num_iterations = 50
        
#         # Get references to the elements we'll animate
#         axes_3d = loss_view[0]  # 3D axes
#         parameter_point = loss_view[3]  # Parameter point on loss surface
        
#         axes_2d = data_view[0]  # 2D axes
#         linear_model = data_view[3]  # Line representing the model
        
#         # Save original positions
#         original_3d_pos = parameter_point.get_center()
        
#         # Run gradient descent with animations
#         for i in range(num_iterations):
#             # Compute gradients
#             grad_theta1 = self.gradient_theta1(theta1, theta2, x_data, y_data)
#             grad_theta2 = self.gradient_theta2(theta1, theta2, x_data, y_data)
            
#             # Update parameters
#             theta1 = theta1 - learning_rate * grad_theta1
#             theta2 = theta2 - learning_rate * grad_theta2
            
#             # Compute current loss
#             current_loss = self.loss_function(theta1, theta2, x_data, y_data)
            
#             # Create new target position for parameter point
#             new_position_3d = axes_3d.c2p(theta1, theta2, current_loss)
            
#             # Create new line for the updated model
#             updated_line_function = lambda x: theta1 * x + theta2
#             new_line = axes_2d.plot(updated_line_function, x_range=[0, 12], color=GREEN)
            
#             # Animate both simultaneously
#             self.play(
#                 parameter_point.animate.move_to(new_position_3d),
#                 Transform(linear_model, new_line),
#                 run_time=0.2 if i > 10 else 0.5  # Speed up later iterations
#             )
            
#             # Add a brief pause at the beginning to see initial state
#             if i == 0:
#                 self.wait(1)
        
#         # Pause at the end to see final state
#         self.wait(2)
    
#     # Loss function: Mean Squared Error
#     def loss_function(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return np.mean(errors**2)
    
#     # Gradient with respect to theta1
#     def gradient_theta1(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return -2 * np.mean(x_data * errors)
    
#     # Gradient with respect to theta2
#     def gradient_theta2(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return -2 * np.mean(errors)


# class GradientDescentOn3DSurface(ThreeDScene):
#     def construct(self):
#         # Create synthetic data
#         np.random.seed(42)  # For reproducibility
#         num_points = 20
#         x_data = np.random.uniform(0, 10, num_points)
#         # True parameters: slope=2, intercept=1
#         true_theta1, true_theta2 = 2.0, 1.0
#         y_data = true_theta1 * x_data + true_theta2 + np.random.normal(0, 1, num_points)
        
#         # Create 3D axes
#         axes = ThreeDAxes(
#             x_range=[-1, 4, 1],
#             y_range=[-1, 4, 1],
#             z_range=[0.0, 10.0, 2.0],
#             height=8,
#             width=8,
#             depth=5,
#             axis_config={
#                 "include_ticks": True,
#                 "color": CHILL_BROWN,
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
#             }
#         )
        
#         # Add axis labels
#         x_label = Text("θ₁", font_size=30).next_to(axes.x_axis.get_end(), RIGHT)
#         y_label = Text("θ₂", font_size=30).next_to(axes.y_axis.get_end(), UP)
#         z_label = Text("Loss", font_size=30).next_to(axes.z_axis.get_end(), OUT)
        
#         # Set up the 3D scene
#         self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
#         self.add(axes, x_label, y_label, z_label)
        
#         # Create the loss surface
#         surface = self.create_loss_surface(axes, x_data, y_data)
#         self.play(FadeIn(surface))
        
#         # Create a parameter point that will move on the surface
#         initial_theta1, initial_theta2 = 0.0, 0.0
#         initial_loss = self.loss_function(initial_theta1, initial_theta2, x_data, y_data)
#         parameter_point = Sphere(radius=0.15, color=ORANGE)
#         parameter_point.move_to(axes.c2p(initial_theta1, initial_theta2, initial_loss))
#         self.play(FadeIn(parameter_point))
        
#         # Run gradient descent with animations
#         self.animate_gradient_descent(axes, parameter_point, x_data, y_data)
        
#         # Final pause
#         self.wait(2)
        
#     def create_loss_surface(self, axes, x_data, y_data):
#         # Function to compute z for any (x, y)
#         def param_surface(u, v):
#             theta1 = u
#             theta2 = v
#             loss = self.loss_function(theta1, theta2, x_data, y_data)
#             return axes.c2p(u, v, loss)
        
#         # Create the surface
#         surface = ParametricSurface(
#             param_surface,
#             u_range=[-1, 4],
#             v_range=[-1, 4],
#             resolution=(30, 30),
#             checkerboard_colors=[BLUE, BLUE.lighter(0.5)],
#             fill_opacity=0.7
#         )
        
#         return surface
    
#     def animate_gradient_descent(self, axes, parameter_point, x_data, y_data):
#         # Initialize parameters
#         theta1, theta2 = 0.0, 0.0
#         learning_rate = 0.01
#         num_iterations = 100
#         path_points = []
        
#         # First iteration outside loop to get starting point
#         current_loss = self.loss_function(theta1, theta2, x_data, y_data)
#         path_points.append(axes.c2p(theta1, theta2, current_loss))
        
#         # Run gradient descent
#         for i in range(num_iterations):
#             # Compute gradients
#             grad_theta1 = self.gradient_theta1(theta1, theta2, x_data, y_data)
#             grad_theta2 = self.gradient_theta2(theta1, theta2, x_data, y_data)
            
#             # Update parameters
#             theta1 = theta1 - learning_rate * grad_theta1
#             theta2 = theta2 - learning_rate * grad_theta2
            
#             # Compute current loss
#             current_loss = self.loss_function(theta1, theta2, x_data, y_data)
            
#             # Save point for path
#             path_points.append(axes.c2p(theta1, theta2, current_loss))
            
#             # Create new target position for parameter point
#             new_position = axes.c2p(theta1, theta2, current_loss)
            
#             # Animate at different speeds depending on iteration
#             if i < 10:
#                 self.play(parameter_point.animate.move_to(new_position), run_time=0.5)
#             elif i < 30:
#                 self.play(parameter_point.animate.move_to(new_position), run_time=0.3)
#             elif i % 5 == 0:  # Only show every 5th step for later iterations
#                 self.play(parameter_point.animate.move_to(new_position), run_time=0.2)
        
#         # Draw the path followed
#         path = VMobject()
#         path.set_points_smoothly(path_points)
#         path.set_color(ORANGE)
#         path.set_stroke(width=3)
#         self.play(ShowCreation(path))
        
#         # Show the final position for longer
#         self.wait(1)
    
#     # Loss function: Mean Squared Error
#     def loss_function(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return np.mean(errors**2)
    
#     # Gradient with respect to theta1
#     def gradient_theta1(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return -2 * np.mean(x_data * errors)
    
#     # Gradient with respect to theta2
#     def gradient_theta2(self, theta1, theta2, x_data, y_data):
#         predictions = theta1 * x_data + theta2
#         errors = y_data - predictions
#         return -2 * np.mean(errors)