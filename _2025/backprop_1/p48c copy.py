from welch_paths import welch_path
import manimgx as m
from welch_scene import manual_camera_interpolation
from landscape_geometry import get_pivot_and_scale
from landscape_geometry import get_numerical_gradient
import numpy as np
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from functools import partial
from tqdm import tqdm

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

wormhole_dir=welch_path('backpropagation/hackin/apr_29_2/')
alphas_1=np.linspace(-2.5, 2.5, 512)

class Dot3D(m.Sphere):
    def __init__(self, center=m.ORIGIN, radius=0.05, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.move_to(center)

def param_surface_2(u, v, surf_array):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = 0.14*surf_array[v_idx, u_idx]-0.07*np.mean(surf_array) #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])




class P48cV3(WelchScene):
    def construct(self):
        '''
        Wikitext training example - man I hope this works. 

        Ok so I think I want to start our here from the same angle etc as the 
        Paris example -> might even be nice to Show Creation or something here. 
        '''
    
        #Load up other surfaces to visualize
        loss_arrays_pre=[]
        loss_arrays_post=[]
        loss_arrays_interleaved=[]
        num_time_steps=4
        print('Loading Surface Arrays...')
        for i in tqdm(range(num_time_steps)):
            loss_arrays_pre.append(np.load(wormhole_dir+'pre_step_'+str(i).zfill(3)+'.npy'))
            loss_arrays_post.append(np.load(wormhole_dir+'post_step_'+str(i).zfill(3)+'.npy'))
            loss_arrays_interleaved.append(loss_arrays_pre[-1])
            loss_arrays_interleaved.append(loss_arrays_post[-1])


        self.wait()
        import matplotlib.pyplot as plt
        data_max=np.array(loss_arrays_interleaved).max() 
        for i in range(len(loss_arrays_interleaved)):
            plt.clf()
            plt.figure(frameon=False)
            ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
            ax.set_axis_off()
            plt.gcf().add_axes(ax)
            plt.imshow(np.rot90(loss_arrays_interleaved[i].T), vmax=1.2*data_max) #Artificial color ceiling to make it not all yellow
            plt.savefig(wormhole_dir+'loss_arrays_interleaved_'+str(i).zfill(3)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()


        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(len(loss_arrays_interleaved))):
            surf_func=partial(param_surface_2, surf_array=loss_arrays_interleaved[i])
            surf_functions.append(surf_func)
            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5],
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = textured_surface(surface, wormhole_dir+'loss_arrays_interleaved_'+str(i).zfill(3)+'.png')
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


        starting_coords=[0.05,-0.9]
        starting_point=surf_functions[0](*starting_coords)
        s2=Dot3D(center=starting_point, radius=0.06, color='$FF00FF')

        self.set_view(180, 23, 0, (-0.06, 0.09, 0.43), 5.81) #Fixed FoV that we end Paris sequence at.
        
        self.wait()
        # self.add(surfaces[0], grids[0], s2)


        self.play(m.Create(surfaces[0]),
                  m.Create(grids[0]), run_time=6.0)
        self.add(s2)
        self.wait()

        # self.play(self.animate_view(137, 47, 0, (0.26, 0.31, -0.15), 5.61), run_time=4)
        self.play(self.animate_view(135, 47, 0, (0.15, 0.28, -0.04), 5.61), run_time=4)
        self.wait()

        self.play(self.animate_view(143, 52, 0, (0.0, -0.33, 0.52), 3.09), run_time=4)
        self.wait()
        
        for i in range(1, len(surfaces)):
            self.remove(surfaces[i-1])
            self.remove(grids[i-1])
            self.add(surfaces[i])
            self.add(grids[i])
            new_point_coords=surf_functions[i](*starting_coords)
            s2.move_to(new_point_coords) 
            self.wait()

        # self.play(self.animate_view(135, 47, 0, (0.15, 0.28, -0.04), 5.61), run_time=4)
        self.play(self.animate_view(180, 23, 0, (-0.06, 0.09, 0.43), 5.81), run_time=4) #v3
        self.wait()

        #Replay from this perspective for some nice side by side. 
        self.remove(surfaces[i])
        self.remove(grids[i])
        self.add(surfaces[0])
        self.add(grids[0])
        new_point_coords=surf_functions[0](*starting_coords)
        s2.move_to(new_point_coords) 

        for i in range(1, len(surfaces)):
            self.remove(surfaces[i-1])
            self.remove(grids[i-1])
            self.add(surfaces[i])
            self.add(grids[i])
            new_point_coords=surf_functions[i](*starting_coords)
            s2.move_to(new_point_coords) 
            self.wait()



        # surface_update_counter=2
        # self.remove(surfaces[surface_update_counter-1])
        # self.remove(grids[surface_update_counter-1])
        # self.add(surfaces[surface_update_counter])
        # self.add(grids[surface_update_counter])
        # new_point_coords=surf_functions[surface_update_counter](*starting_coords)
        # s2.move_to(new_point_coords) 
        # self.wait()



        #Todo -> come back and work out views & time step stuff - depends a bit on where the main p48 ends too. 


        # num_total_steps=38 #Crank this for final viz
        # start_orientation=[142, 34, 0, (-0.09, -0.77, 0.15), 3.55]
        # # end_orientation=[131, 31, 0, (-0.12, -0.88, 0.22), 2.90]
        # end_orientation=[122, 11, 0, (-0.01, -0.28, 0.55), 2.05] #Move overhead
        # interp_orientations=manual_camera_interpolation(start_orientation, end_orientation, num_steps=num_total_steps)

        # surface_update_counter=1
        # frames_per_surface_upddate=np.floor(num_total_steps/num_time_steps)
        # self.wait()
        # for i in range(1, num_total_steps):
        #     # print(i, len(interp_orientations))
        #     if i%frames_per_surface_upddate==0 and surface_update_counter<len(surfaces):

        #         self.remove(surfaces[surface_update_counter-1])
        #         self.remove(grids[surface_update_counter-1])
        #         self.add(surfaces[surface_update_counter])
        #         self.add(grids[surface_update_counter])

        #         new_point_coords=surf_functions[surface_update_counter](*starting_coords)
        #         s2.move_to(new_point_coords) #This should make point move down smoothly. 
        #         surface_update_counter+=1

        #     # print(i, len(interp_orientations))
        #     self.set_view(*interp_orientations[i])
        #     self.wait(0.1)

        self.wait(20)
        pass