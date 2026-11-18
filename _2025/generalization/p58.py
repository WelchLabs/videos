from welch_paths import welch_path
import manimgx as gx
from welch_geometry import diagram_axes_3d
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from welch_text import math_tex

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#
CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

class NeuralNetwork(nn.Module):
    def __init__(self, activation_function=nn.Sigmoid()):
        super(NeuralNetwork, self).__init__()
        
        # Layer 1: 2 inputs -> 2 hidden units
        self.w1 = nn.Parameter(torch.randn(2, 2))  # 2x2 weight matrix
        self.b1 = nn.Parameter(torch.randn(2))     # 2 biases
        
        # Layer 2: 2 hidden units -> 1 output
        self.w2 = nn.Parameter(torch.randn(2, 1))  # 2x1 weight matrix
        self.b2 = nn.Parameter(torch.randn(1))     # 1 bias
        
        self.activation_function = activation_function

    def forward(self, x):
        # First layer
        h = self.activation_function(torch.matmul(x, self.w1) + self.b1)
        # Second layer
        y = self.activation_function(torch.matmul(h, self.w2) + self.b2)
        return y

#Training data for 0/1 
X = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

class StepFunction(nn.Module):
    def forward(self, x):
        return torch.where(x >= 0, 1.0, 0.0)


class NeuralNetworkTwoParam(nn.Module):
    def __init__(self, activation_function=nn.Sigmoid()):
        super(NeuralNetworkTwoParam, self).__init__()
        
        # Only w10 as Parameter - will be optimized
        # self.w10 = nn.Parameter(torch.tensor([1, 1], dtype=torch.float32))
        self.w10 = nn.Parameter(torch.tensor([-3, -1.5], dtype=torch.float32)) #High error values starting point
        
        # All other tensors as regular tensors - will not be optimized
        self.w11 = torch.tensor([1, 1], dtype=torch.float32)
        self.b1 = torch.tensor([0.5, -1.5], dtype=torch.float32)
        self.w2 = torch.tensor([[1], [-1]], dtype=torch.float32)
        self.b2 = torch.tensor([-0.4], dtype=torch.float32)
        
        self.activation_function = activation_function

    def forward(self, x):
        # First layer
        w1 = torch.vstack((self.w10, self.w11))
        h = self.activation_function(torch.matmul(x, w1) + self.b1)
        # Second layer
        y = self.activation_function(torch.matmul(h, self.w2) + self.b2)
        return y

class P66v3(WelchScene):
    def construct(self):
        # Create the same model and compute losses as before
        model = NeuralNetwork(StepFunction())
        criterion = nn.MSELoss()
        
        # Set up the model parameters
        model.w1 = torch.nn.Parameter(torch.tensor([[1,1],[1,1]], dtype=torch.float32))
        model.b1 = torch.nn.Parameter(torch.tensor([0.5, -1.5], dtype=torch.float32))
        model.w2 = torch.nn.Parameter(torch.tensor([[1],[-1]], dtype=torch.float32))
        model.b2 = torch.nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
        
        # Compute losses (your existing code)
        all_losses = []
        all_losses_batch = []
        x_range = np.linspace(-4, 4, 256)
        y_range = np.linspace(-4, 4, 256)
        
        with torch.no_grad():
            for v1 in tqdm(x_range):
                for v2 in y_range:
                    model.w1 = torch.nn.Parameter(torch.tensor([[v1,v2],[1,1]], dtype=torch.float32))
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    all_losses_batch.append(loss.item())
        all_losses_batch = np.array(all_losses_batch).reshape(256, 256)
        
        all_losses_batch = all_losses_batch*2  # Quick hacky scaling thing
          
       # Create the surface
        axes = diagram_axes_3d(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[0, 2.0, 0.5],
            height=8,
            width=8,
            depth=3,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )
        
        # Add labels
        x_label = math_tex('w_{00}', font_size=28).set_color(CHILL_BROWN)
        y_label = math_tex('w_{01}', font_size=28).set_color(CHILL_BROWN)
        z_label = math_tex('Error^2', font_size=28).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, gx.RIGHT)
        y_label.next_to(axes.y_axis, gx.UP)
        z_label.next_to(axes.z_axis, gx.OUT)
        z_label.rotate(90*gx.DEGREES, [1,0,0])
    
        def param_surface(u, v):
            x = u
            y = v
            x_idx = int((x + 4) * 256/8)
            y_idx = int((y + 4) * 256/8)
            x_idx = np.clip(x_idx, 0, 255)
            y_idx = np.clip(y_idx, 0, 255)
            z = all_losses_batch[x_idx, y_idx]
            return np.array([x, y, z])

        # Create main surface
        surface = parametric_surface(
            param_surface,
            u_range=[-4, 4],
            v_range=[-4, 4],
            resolution=(256, 256),
        )
        
        # Create the textured surface
        ts = textured_surface(surface, 'step_function_activation_landscape_1.png')
        set_surface_shading(ts, 0.0, 0.1, 0)
        ts.set_opacity(0.7)
        
        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 256  # Number of points per line
        u_gridlines = gx.VGroup()
        v_gridlines = gx.VGroup()
        
        # Create u-direction gridlines
        u_values = np.linspace(-4, 4, num_lines)
        v_points = np.linspace(-4, 4, num_points)
        
        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = gx.VMobject()
            line.set_points_as_corners(points)
            line.set_stroke(width=1, color=gx.WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-4, 4, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface(u, v) for u in u_points]
            line = gx.VMobject()
            line.set_points_as_corners(points)
            line.set_stroke(width=1, color=gx.WHITE, opacity=0.3)
            v_gridlines.add(line)
        
        # Add everything to the scene
        self.set_view(0, 0, 0, (0.05, -0.06, 0.0), 12)
        self.add(ts)
        self.add(u_gridlines, v_gridlines)
        self.add(axes, x_label, y_label, z_label)
        self.wait(1)

        self.play(self.animate_view(35, 57, 0, (-0.02, -0.09, 0.03), 10.69), run_time=4)
        self.wait()
        self.play(self.animate_view(137, 56, 0, (-0.02, -0.09, 0.03), 10.69), run_time=16, rate_func=gx.linear)
        self.wait()

        #One more move at the same angles as the otehr one. 
        self.play(self.animate_view(26, 45, 0, (-0.41, 0.51, 1.03), 10.85), run_time=4)
        self.wait()
        self.play(self.animate_view(122, 66, 0, (-0.41, 0.51, 1.03), 10.85), run_time=4)
        self.wait()

        # reorient(26, 45, 0, (-0.41, 0.51, 1.03), 10.85)


        self.wait(20)
        

        # self.embed()     

class P58(WelchScene):
    def construct(self):

        model = NeuralNetworkTwoParam(nn.Sigmoid())
        criterion = nn.MSELoss()
        # optimizer = optim.SGD(model.parameters(), lr=10.0)

        all_losses_batch = []
        x_range = np.linspace(-4, 4, 256)
        y_range = np.linspace(-4, 4, 256)
        
        with torch.no_grad():
            for v1 in tqdm(x_range):
                for v2 in y_range:
                    model.w10=torch.nn.Parameter(torch.tensor([v1,v2], dtype=torch.float32))
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    all_losses_batch.append(loss.item())
        all_losses_batch = np.array(all_losses_batch).reshape(256, 256)
        
        all_losses_batch = all_losses_batch*50  # Quick hacky scaling thing
        all_losses_batch=all_losses_batch-np.min(all_losses_batch)

        model = NeuralNetworkTwoParam(nn.Sigmoid())
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=3.0)

        history=[]
        for i in range(500):
            #Stochastic
            # outputs = model(X[i%len(y)]) #Hmm do i want batch gradient descent? Probably no. 
            # loss = criterion(outputs, y[i%len(y)])

            outputs = model(X) #Hmm do i want batch gradient descent? Probably no. 
            loss = criterion(outputs, y)
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append([i, loss.item(), *model.w10.detach().numpy().tolist()])

        history=np.array(history)

        # Create the surface
        axes = diagram_axes_3d(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[0, 2.0, 0.5],
            height=8,
            width=8,
            depth=3,
            axis_config={
                "include_ticks": True,
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_tip": True,
                "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
            }
        )
        
        # Add labels
        x_label = math_tex('w_{00}', font_size=28).set_color(CHILL_BROWN)
        y_label = math_tex('w_{01}', font_size=28).set_color(CHILL_BROWN)
        z_label = math_tex('Error^2', font_size=28).set_color(CHILL_BROWN)
        x_label.next_to(axes.x_axis, gx.RIGHT)
        y_label.next_to(axes.y_axis, gx.UP)
        z_label.next_to(axes.z_axis, gx.OUT)
        z_label.rotate(90*gx.DEGREES, [1,0,0])
    
        def param_surface(u, v):
            x = u
            y = v
            x_idx = int((x + 4) * 256/8)
            y_idx = int((y + 4) * 256/8)
            x_idx = np.clip(x_idx, 0, 255)
            y_idx = np.clip(y_idx, 0, 255)
            z = all_losses_batch[x_idx, y_idx]
            return np.array([x, y, z])

        # Create main surface
        surface = parametric_surface(
            param_surface,
            u_range=[-4, 4],
            v_range=[-4, 4],
            resolution=(256, 256),
        )
        
        # Create the textured surface
        ts = textured_surface(surface, welch_path('double_descent/graphics/sigmoid_function_activation_landscape_1.png'))
        set_surface_shading(ts, 0.0, 0.1, 0)
        ts.set_opacity(0.7)
        
        # Create gridlines using polylines instead of parametric curves
        num_lines = 20  # Number of gridlines in each direction
        num_points = 256  # Number of points per line
        u_gridlines = gx.VGroup()
        v_gridlines = gx.VGroup()
        
        # Create u-direction gridlines
        u_values = np.linspace(-4, 4, num_lines)
        v_points = np.linspace(-4, 4, num_points)
        
        for u in u_values:
            points = [param_surface(u, v) for v in v_points]
            line = gx.VMobject()
            line.set_points_as_corners(points)
            line.set_stroke(width=1, color=gx.WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-4, 4, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface(u, v) for u in u_points]
            line = gx.VMobject()
            line.set_points_as_corners(points)
            line.set_stroke(width=1, color=gx.WHITE, opacity=0.3)
            v_gridlines.add(line)
        

         # Create the gradient descent path
        path_points = []
        for step in history: #[::25]:
            _, loss, w0, w1 = step
            x_idx = int((w0 + 4) * 256/8)
            y_idx = int((w1 + 4) * 256/8)
            x_idx = np.clip(x_idx, 0, 255)
            y_idx = np.clip(y_idx, 0, 255)
            z = all_losses_batch[x_idx, y_idx]
            path_points.append([w0, w1, z])
        
        # Create segments for the path with fading opacity
        path_segments = gx.Group()
        for i in range(len(path_points) - 1):
            start = np.array(path_points[i])
            end = np.array(path_points[i + 1])
            
            # Create line segment
            line = gx.Line3D(
                start=start,
                end=end,
                width=0.05,
                #stroke_color=WHITE,
                #stroke_opacity=min(0.8, 0.1 + i/len(path_points))  # Opacity increases along path
            )
            line.set_color(gx.WHITE)
            path_segments.add(line)
        
        # Add start and end points markers
        start_point = gx.Sphere(radius=0.05, color=YELLOW)
        start_point.move_to(path_points[0])
        
        end_point = gx.Sphere(radius=0.125, color=gx.RED)
        end_point.move_to(path_points[0])

        # Add everything to the scene
        # self.set_view(0, 0, 0, (0.05, -0.06, 0.0), 12)
        self.add(ts)
        self.add(u_gridlines, v_gridlines)
        self.add(axes, z_label) #, x_label, y_label, z_label)
        self.wait(1)


        self.set_view(137, 56, 0, (-0.02, -0.09, 0.03), 10.69) #Where p66 left of I think


        
        self.wait(1)

        self.play(self.animate_view(26, 45, 0, (-0.41, 0.51, 1.03), 10.85), run_time=10)
        self.wait()
        self.add(end_point)
        self.wait()

        #Now rotate around as optimizer runs 
        total_points=len(path_segments)
        point_count=0
        initial_orientation = (26, 45, 0)
        final_orientation=(122, 66, 0)

        # reorient(26, 45, 0, (-0.41, 0.51, 1.03), 10.85)

        for i in range(len(path_segments)):
            self.remove(end_point)
            self.add(path_segments[i])
            end_point.move_to(path_points[i])
            self.add(end_point)
            progress = point_count / total_points
            current_theta = initial_orientation[0] + (final_orientation[0] - initial_orientation[0]) * progress
            current_phi = initial_orientation[1] + (final_orientation[1] - initial_orientation[1]) * progress
            current_gamma = initial_orientation[2] + (final_orientation[2] - initial_orientation[2]) * progress
            
            # Update camera position
            self.set_view(current_theta, current_phi, current_gamma, (-0.41, 0.51, 1.03), 10.85)
            self.wait(1./30)   
            point_count += 1    

        self.wait()

        # Calculate total number of points for camera movement interpolation
        # w0_range = np.arange(-1.9, 2.0, 0.25) #CRANK UP DENSITY FOR FINAL VIZ, AND MAKE SURE IT MATCHES DIAL TURNS ABOVE
        # w1_range = np.arange(-1.9, 2.0, 0.25)
        # total_points = len(w0_range) * len(w1_range)
        # point_count = 0

        # # Set initial and final camera orientations
        # initial_orientation = (-29, 59, 0)
        # final_orientation = (59, 68, 0)

        # for w0 in w0_range:
        #     for w1 in w1_range:
        #         yhat = X[:,0]*w0 + X[:,1]*w1 + b
        #         error = np.mean((y.ravel()-yhat)**2)
        #         point = axes3d.c2p(w0, w1, error)
        #         dot = Sphere(radius=0.05, color=BLUE, opacity=1).move_to(point)
        #         dots.add(dot)
                
        #         # Calculate interpolated camera angles
        #         progress = point_count / total_points
        #         current_theta = initial_orientation[0] + (final_orientation[0] - initial_orientation[0]) * progress
        #         current_phi = initial_orientation[1] + (final_orientation[1] - initial_orientation[1]) * progress
        #         current_gamma = initial_orientation[2] + (final_orientation[2] - initial_orientation[2]) * progress
                
        #         # Update camera position
        #         self.set_view(current_theta, current_phi, current_gamma, (-0.11, 0.15, 1.36), 8.00) #This is looking dope!



        # self.add(path_segments)
        self.wait()

        self.wait(20)






# class P66(ThreeDScene):
#     def construct(self):
#         # Create the same model and compute losses as before
#         model = NeuralNetwork(StepFunction())
#         criterion = nn.MSELoss()
        
#         # Set up the model parameters
#         model.w1 = torch.nn.Parameter(torch.tensor([[1,1],[1,1]], dtype=torch.float32))
#         model.b1 = torch.nn.Parameter(torch.tensor([0.5, -1.5], dtype=torch.float32))
#         model.w2 = torch.nn.Parameter(torch.tensor([[1],[-1]], dtype=torch.float32))
#         model.b2 = torch.nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))

#         # Compute losses (your existing code)
#         all_losses = []
#         all_losses_batch = []
#         x_range = np.linspace(-4, 4, 256)
#         y_range = np.linspace(-4, 4, 256)
        
#         with torch.no_grad():
#             for v1 in tqdm(x_range):
#                 for v2 in y_range:
#                     model.w1 = torch.nn.Parameter(torch.tensor([[v1,v2],[1,1]], dtype=torch.float32))
#                     outputs = model(X)
#                     loss = criterion(outputs, y)
#                     all_losses_batch.append(loss.item())

#         all_losses_batch = np.array(all_losses_batch).reshape(256, 256)
        
#         all_losses_batch=all_losses_batch*2 #Quick hacky scaling thing

#         # Create the surface
#         axes = ThreeDAxes(
#             x_range=[-4, 4, 1],
#             y_range=[-4, 4, 1],
#             z_range=[0, 2.0, 0.5],
#             height=8,
#             width=8,
#             depth=3,
#             axis_config={
#                 "include_ticks": True,
#                 "color": CHILL_BROWN,
#                 "stroke_width": 2,
#                 "include_tip": True,
#                 "tip_config": {"fill_opacity": 1, "width": 0.1, "length": 0.1}
#             }
#         )

#         # Add labels
#         x_label = Tex('w_{00}', font_size=28).set_color(CHILL_BROWN)
#         y_label = Tex('w_{01}', font_size=28).set_color(CHILL_BROWN)
#         z_label = Tex('Error^2', font_size=28).set_color(CHILL_BROWN)
#         x_label.next_to(axes.x_axis, RIGHT)
#         y_label.next_to(axes.y_axis, UP)
#         z_label.next_to(axes.z_axis, OUT)
#         z_label.rotate(90*DEGREES, [1,0,0])

    
#         # Create surface function
#         def param_surface(u, v):
#             x = u
#             y = v
#             # Find closest indices in our computed loss array
#             x_idx = int((x + 4) * 256/8)
#             y_idx = int((y + 4) * 256/8)
#             x_idx = np.clip(x_idx, 0, 255)
#             y_idx = np.clip(y_idx, 0, 255)
#             z = all_losses_batch[x_idx, y_idx]
#             return np.array([x, y, z])

#         # Create the surface with color gradient
#         surface = ParametricSurface(
#             param_surface,
#             u_range=[-4, 4],
#             v_range=[-4, 4],
#             resolution=(50, 50),
#             #checkerboard_colors=[BLUE_D, BLUE_E],
#             #stroke_width=0.5
#         )

#         ts=TexturedSurface(surface, 'step_function_activation_landscape_1.png')
#         set_surface_shading(ts, 0.0,0.2,0)
#         # set_surface_shading(ts, 0,1,0)
#         # set_surface_shading(ts, 0,0d,1)
#         ts.set_opacity(0.7)

#         self.add(ts)
#         self.add(axes, x_label, y_label, z_label)

#         self.wait(2)




# class P66(InteractiveScene):
#     def construct(self):

#         model = NeuralNetwork(StepFunction())
#         # criterion = nn.BCELoss()  # Binary Cross Entropy Loss
#         criterion = nn.MSELoss()
#         optimizer = optim.SGD(model.parameters(), lr=0.1)

#         model.w1=torch.nn.Parameter(torch.tensor([[1,1],[1,1]], 
#                              dtype=torch.float32))
#         model.b1=torch.nn.Parameter(torch.tensor([0.5, -1.5], 
#                                      dtype=torch.float32))
#         model.w2=torch.nn.Parameter(torch.tensor([[1],[-1]], 
#                                      dtype=torch.float32))
#         model.b2=torch.nn.Parameter(torch.tensor([-0.5], 
#                                      dtype=torch.float32))

#         all_losses=[]
#         all_losses_batch=[]
#         with torch.no_grad():
#             for v1 in tqdm(np.linspace(-4, 4, 256)):
#                 for v2 in np.linspace(-4, 4, 256):
#                     # model.b1=torch.nn.Parameter(torch.tensor([b1,b2], dtype=torch.float32))
#                     model.w1=torch.nn.Parameter(torch.tensor([[v1,v2],[1,1]], 
#                                      dtype=torch.float32))
#                     #Batch
#                     outputs=model(X)
#                     loss = criterion(outputs, y)
#                     batch_loss=loss.item()
#                     all_losses_batch.append(batch_loss)
                    
#                     #Stochastic
#                     losses_by_example=[]
#                     for i in range(4):
#                         outputs=model(X[i])
#                         loss=criterion(outputs, y[i])
#                         losses_by_example.append(loss.item())
            
#                     all_losses.append(losses_by_example)
                
#         all_losses=np.array(all_losses); all_losses_batch=np.array(all_losses_batch)
#         all_losses_batch=all_losses_batch.reshape(256,256)