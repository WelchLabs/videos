from welch_paths import welch_path
import numpy as np
import manimgx as m
from landscape_geometry import get_numerical_gradient
from welch_geometry import diagram_axes_3d
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from functools import partial
from tqdm import tqdm

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

save_dir=welch_path('backpropagation/hackin/')
xy=np.linspace(-2.5, 2.5, 256)

import numpy as np
import matplotlib.pyplot as plt

# Create a grid of x and y values
x = np.linspace(-2.5, 2.5, 256)
y = np.linspace(-2.5, 2.5, 256)
X, Y = np.meshgrid(x, y)
Z = 0.25 * (X**2 + Y**2)

plt.figure(frameon=False)
ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
ax.set_axis_off()
plt.gcf().add_axes(ax)
plt.imshow(np.rot90(Z)) #have to transpose if transposing u and v and param_surface_1
plt.savefig(save_dir+'thumnbail_simple_loss_1.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()


wormhole_dir=welch_path('backpropagation/hackin/wormhole_merged/')
alphas_1=np.linspace(-2.5, 2.5, 512)
loss_2d_1=np.load(wormhole_dir+'000.npy')


def param_surface_1(u, v):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = 0.07*loss_2d_1[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])

def param_surface_2(u, v, surf_array):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = 0.07*surf_array[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])


class Dot3D(m.Sphere):
    def __init__(self, center=m.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)

class ThumbnailOneV2(WelchScene):
    def construct(self):
        '''
        Simple 2d grad descent to juxpopose with 1.2B dimensinoal descent.      
        '''

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

        def param_surface(u, v):
            u_idx = np.abs(xy - u).argmin()
            v_idx = np.abs(xy - v).argmin()
            try:
                z = Z[u_idx, v_idx]
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
        
        ts = textured_surface(surface, save_dir+'thumnbail_simple_loss_1.png')
        set_surface_shading(ts, 0.0, 0.1, 0)
        ts.set_opacity(1.0)

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
            line.set_stroke(width=2, color=m.WHITE, opacity=0.3)
            u_gridlines.add(line)
        
        # Create v-direction gridlines
        u_points = np.linspace(-2.5, 2.5, num_points)
        for v in u_values:  # Using same number of lines for both directions
            points = [param_surface(u, v) for u in u_points]
            line = m.VMobject()
            # line.set_points_as_corners(points)
            line.set_points_smoothly(points)
            line.set_stroke(width=2, color=m.WHITE, opacity=0.3)
            v_gridlines.add(line)

        #i think there's a better way to do this
        offset=surface.get_corner((m.config.frame_height / 2 * m.DOWN)+m.LEFT)-axes.get_corner((m.config.frame_height / 2 * m.DOWN)+m.LEFT)
        axes.shift(offset);
        
        # Add everything to the scene
        # self.set_view(0, 27, 0, (-0.42, 0.55, 0.61), 7.48)
        self.set_view(0, 0, 0, (0,0,0), 8.0)
        # self.add(axes[:2])#, z_label) #Don't really need vertical axis here!
        self.add(ts, u_gridlines, v_gridlines)
        self.wait()



        starting_coords=[-2.2,-1.5] 
        starting_point=param_surface(*starting_coords)
        s1=Dot3D(center=starting_point, radius=0.17, color=YELLOW)
        # self.add(s1)


        num_steps=128 # I think it gest stuck around 30-40 at lr -0.01 - play with this for smoother/longer animation
        learning_rate=3e-3
        momentum = 0.95 #lil momentum to get to a decent minima lol
        trajectory=[[starting_point[0], starting_point[1], param_surface(starting_point[0], starting_point[1])[2]]]
        velocity = np.array([0.0, 0.08]) #np.zeros(2)
        for i in range(num_steps):
            g=get_numerical_gradient(param_surface, trajectory[-1][0], trajectory[-1][1], epsilon=0.01)
            velocity = momentum * velocity - learning_rate * np.array(g)
            new_x = trajectory[-1][0] + velocity[0]
            new_y = trajectory[-1][1] + velocity[1]
            trajectory.append([new_x, new_y, param_surface(new_x, new_y)[2]])

        t = m.VMobject()
        t.set_stroke(width=10, color=YELLOW, opacity=0.9)
        t.set_points_smoothly(trajectory)
        self.add(t)

        # for i in range(num_steps): #Go partially and then add countour
        #     # s1.move_to(trajectory[i])
        #     t.set_points_smoothly(trajectory[:i])
            # self.wait(0.1)
        s1.move_to(trajectory[-1])
        self.add(s1)
        self.wait()
        


        # self.wait(20)
        pass


class ThumbnailTwoV1(WelchScene):
    def construct(self):
        starting_coords=[0.05,-0.9]
        starting_point=param_surface_1(*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        num_time_steps=1
        print('Loading Surface Arrays')
        for i in tqdm(range(num_time_steps)):
            loss_arrays.append(np.load(wormhole_dir+str(i).zfill(3)+'.npy'))

        # import matplotlib.pyplot as plt
        # for i in range(num_time_steps):
        #     plt.clf()
        #     plt.figure(frameon=False)
        #     ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        #     ax.set_axis_off()
        #     plt.gcf().add_axes(ax)
        #     plt.imshow(np.rot90(loss_arrays[i].T)) #have to transpose if transposing u and v and param_surface_1
        #     plt.savefig(wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #     plt.close()


        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(num_time_steps)):
            surf_func=partial(param_surface_2, surf_array=loss_arrays[i])
            surf_functions.append(surf_func)
            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = textured_surface(surface, wormhole_dir+'loss_2d_1_'+str(i).zfill(3)+'.png')
            set_surface_shading(ts2, 0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number of points per line
            u_gridlines = m.VGroup()
            v_gridlines = m.VGroup()
            u_values = np.linspace(-2.5, 2.5, num_lines)
            v_points = np.linspace(-2.5, 2.5, num_points)
            for u in u_values:
                points = [surf_func(u, v) for v in v_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                u_gridlines.add(line)

            u_points = np.linspace(-2.5, 2.5, num_points)
            for v in u_values:  # Using same number of lines for both directions
                points = [surf_func(u, v) for u in u_points]
                line = m.VMobject()
                line.set_points_smoothly(points)
                line.set_stroke(width=1, color=m.WHITE, opacity=0.15)
                v_gridlines.add(line)
            grids.add(m.VGroup(u_gridlines, v_gridlines))

        self.wait()

        self.add(surfaces[0])
        self.add(grids[0])
        self.add(s2)
        self.remove(s2)

        # self.remove(surfaces[0])

        self.set_view(137, 20, 0, (0.15, 0.11, -0.02), 7.95)
        self.wait()

        self.set_view(137, 20, 0, (0.15, 0.11, -0.02), 7.95)
        self.wait()

        self.set_view(93, 41, 0, (-0.03, 0.01, 0.15), 6.00)
        self.wait()

        self.set_view(0, 21, 0, (-0.03, 0.0, 0.12), 6.32)
        self.wait()

        self.set_view(0, 42, 0, (0.06, 0.13, 0.25), 6.32)
        self.wait()



        #Fixed orentation
        # self.set_view(132, 28, 0, (-0.12, -0.56, 0.33), 4.50) #Kinda wide, but nice I think, could do a closer one too
        # self.set_view(159, 35, 0, (-0.03, -0.62, 0.65), 1.95) #Match with long render
        # self.set_view(180, 23, 0, (-0.06, 0.09, 0.43), 5.81) #Fixed wide veiew I can use at the beginning of 48
        # self.set_view(135, 47, 0, (0.15, 0.28, -0.04), 5.61) #V2 View for side-by-side in p49


        self.wait()
        pass

        # surface_update_counter=1
        # frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        # self.wait()
        # for i in range(1, len(surfaces)):
        #     # print(i, len(interp_orientations))

        #     self.remove(surfaces[surface_update_counter-1])
        #     self.remove(grids[surface_update_counter-1])
        #     self.add(surfaces[surface_update_counter])
        #     self.add(grids[surface_update_counter])
        #     # print('surface_update_counter', surface_update_counter)

        #     new_point_coords=surf_functions[surface_update_counter](*starting_coords)
        #     s2.move_to(new_point_coords) #This should make point move down smoothly. 
        #     surface_update_counter+=1
        #     self.wait(0.1)

        # self.wait()

        # self.wait()
        # self.play(self.animate_view(-103, 12, 0, (0.01, -0.46, 0.57), 1.95), run_time=10.0) #Pan Around
        # self.wait()
        # self.play(self.animate_view(-89, 0, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Zoom out
        # self.wait()
        # self.play(self.animate_view(-85, 99, 0, (0.05, -0.09, 0.59), 5.82), run_time=8.0) #Pan below
        # self.wait()
        # self.play(self.animate_view(84, 102, 0, (0.05, -0.09, 0.59), 5.82), run_time=10.0) #Pan around below
        # self.wait()
        # self.play(self.animate_view(89, 0, 0, (-0.03, -0.14, 0.51), 6.38), run_time=8.0) #Back to overheaad
        # self.wait()

        self.wait(20)
        pass



