from welch_paths import welch_path
import manimgx as m
from landscape_geometry import get_pivot_and_scale
import numpy as np
from welch_scene import WelchScene
from welch_surface import parametric_surface, set_surface_shading, textured_surface
from functools import partial
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'

MAX_RENDERS=100

data_dir=welch_path('backpropagation/poster/gpu_renders_4/')
alphas_1=np.linspace(-2.5, 2.5, 512)

##Only need to run this when the underlying npy file changes
# import matplotlib.pyplot as plt
# plt.figure(frameon=False)
# ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
# ax.set_axis_off()
# plt.gcf().add_axes(ax)
# plt.imshow(np.rot90(loss_2d_1.T)) #have to transpose if transposing u and v and param_surface_1
# plt.savefig(wormhole_dir+'loss_2d_1.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# plt.clf()
# plt.figure(frameon=False)
# ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
# ax.set_axis_off()
# plt.gcf().add_axes(ax)
# plt.imshow(np.rot90(loss_2d_1.T)[128:-128, 128:-128])
# plt.savefig('loss_2d_1_inner.png', bbox_inches='tight', pad_inches=0, dpi=300)
# plt.close()

# Comman for rendering all scenes
# manimgx render /Users/stephen/manim/videos/_2025/backprop_1/p39_48.py P39_47 P48_moving_view_1 P48_moving_view_2 P48_moving_view_3 P48_fixed_view -w


def param_surface_2(u, v, surf_array):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = 0.07*surf_array[v_idx, u_idx] #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])

def param_surface_3(u, v, surf_array, scaling=0.07, surf_mean=0):
    u_idx = np.abs(alphas_1 - u).argmin()
    v_idx = np.abs(alphas_1 - v).argmin()
    try:
        # z = loss_2d_1[u_idx, v_idx]
        z = scaling*(surf_array[v_idx, u_idx]-surf_mean) #Add vertical scaling here?
    except IndexError:
        z = 0
    return np.array([u, v, z])



class PosterHackinEight(WelchScene):
    def construct(self):
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        paths=[]
        texture_paths=[]
        print('Loading Surface Arrays...')

        render_paths=glob.glob(data_dir+'*')
        for r in render_paths:
        # r=render_paths[0]
            for np_path in glob.glob(r+'/*.npy'):
                # print(np_path)
                loss_arrays.append(np.load(np_path))
                print(np_path, loss_arrays[-1].shape)
                paths.append(np_path)
                plt.clf()
                plt.figure(frameon=False)
                ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
                ax.set_axis_off()
                plt.gcf().add_axes(ax)
                plt.imshow(np.rot90(loss_arrays[-1].T)) #have to transpose if transposing u and v and param_surface_1
                texture_paths.append(r+'/'+np_path.split('/')[-1].split('.')[0]+'_texture.png')
                plt.savefig(texture_paths[-1], bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close()

        self.wait()

        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        texts=m.Group()
        print("Loading Surfaces and Gridlines...")
        for i in tqdm(range(len(paths[:MAX_RENDERS]))): #LIMIT HERE FOR TESTING
            if 'llama' in paths[i]: scaling=0.07
            elif 'gpt' in paths[i]: scaling=0.2
            elif 'qwen' in paths[i]: scaling=0.07
            elif 'gemma' in paths[i]: scaling=0.07
            else: scaling=0.07

            surf_func=partial(param_surface_3, surf_array=loss_arrays[i], scaling=scaling, surf_mean=np.mean(loss_arrays[i]))
            surf_functions.append(surf_func)
            surface = parametric_surface(
                surf_func,  
                u_range=[-2.5, 2.5], #Some have different ranges, but I don't think it matters?
                v_range=[-2.5, 2.5],
                resolution=(512, 512),
            )

            ts2 = textured_surface(surface, texture_paths[i])
            set_surface_shading(ts2, 0.0, 0.1, 0)
            surfaces.add(ts2)

            num_lines = 64  # Number of gridlines in each direction
            num_points = 512  # Number  of points per line
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

            t=m.Text(paths[i][-40:])
            t.scale(0.4)
            t.rotate(m.DEGREES*180)
            t.move_to([0, 3, 0])
            texts.add(t)

        # self.set_view(134, 30, 0, (0.12, 0.11, -0.15), 7.56)
        # self.set_view(132, 30, 0, (0.32, 0.4, -0.32), 6.82)
        # self.set_view(132, 30, 0, (0.24, 0.27, -0.23), 7.50)
        # self.set_view(137, 41, 0, (0.14, -0.04, -0.09), 6.81) # I think this is what looked pretty good from part 1
        self.set_view(137, 41, 0, (0.48, 0.32, -0.52), 6.81)
        self.add(surfaces[0])
        self.add(grids[0])
        # self.add(texts[0])
        self.wait()

        for i in range(1, np.min([len(paths), MAX_RENDERS])):
            self.remove(surfaces[i-1])
            self.remove(grids[i-1])
            self.remove(texts[i-1])
            self.add(surfaces[i])
            self.add(grids[i])
            # self.add(texts[i])
            self.wait()


        # self.set_view(135, 25, 0, (0.02, 0.1, -0.11), 7.56) #isometric, fiarly high
        # # self.wait()

        # self.set_view(137, 41, 0, (0.14, -0.04, -0.09), 6.81) #isometric, more down
        # self.wait()


        self.wait()
        pass



class SingleTest(WelchScene):
    def construct(self):
    
        #Load up other surfaces to visualize
        loss_arrays=[]
        paths=[]
        texture_paths=[]
        print('Loading Surface Arrays...')

        render_paths=glob.glob(data_dir+'*')
        for r in render_paths:
        # r=render_paths[0]
            for np_path in glob.glob(r+'/*.npy'):
                # print(np_path)
                loss_arrays.append(np.load(np_path))
                print(np_path, loss_arrays[-1].shape)
                paths.append(np_path)
                plt.clf()
                plt.figure(frameon=False)
                ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
                ax.set_axis_off()
                plt.gcf().add_axes(ax)
                plt.imshow(np.rot90(loss_arrays[-1].T)) #have to transpose if transposing u and v and param_surface_1
                texture_paths.append(r+'/'+np_path.split('/')[-1].split('.')[0]+'_texture.png')
                plt.savefig(texture_paths[-1], bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close()

        self.wait()

        surfaces=m.Group()
        surf_functions=[] #Need this later to move dot around.
        grids=m.Group()
        texts=m.Group()
        print("Loading Surfaces and Gridlines...")

        i=-1
        print(paths[i])

        scaling=0.07
        if 'llama' in paths[i]: scaling=0.07
        if 'gpt' in paths[i]: scaling=0.2
        if 'gpt' in paths[i] and 'may_7_1' in paths[i]: scaling=0.02
        if 'qwen' in paths[i]: scaling=0.07
        if 'gemma' in paths[i]: scaling=0.07

        surf_func=partial(param_surface_3, surf_array=loss_arrays[i], scaling=scaling, surf_mean=np.mean(loss_arrays[i]))
        surf_functions.append(surf_func)
        surface = parametric_surface(
            surf_func,  
            u_range=[-2.5, 2.5], #Some have different ranges, but I don't think it matters?
            v_range=[-2.5, 2.5],
            resolution=(512, 512),
        )

        ts2 = textured_surface(surface, texture_paths[i])
        set_surface_shading(ts2, 0.0, 0.1, 0)
        surfaces.add(ts2)

        num_lines = 64  # Number of gridlines in each direction
        num_points = 512  # Number  of points per line
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

        t=m.Text(paths[i][-40:])
        t.scale(0.4)
        t.rotate(m.DEGREES*180)
        t.move_to([0, 3, 0])
        texts.add(t)

        # self.set_view(134, 30, 0, (0.12, 0.11, -0.15), 7.56)
        # self.set_view(132, 30, 0, (0.32, 0.4, -0.32), 6.82)
        # self.set_view(132, 30, 0, (0.24, 0.27, -0.23), 7.50)
        # self.set_view(137, 41, 0, (0.14, -0.04, -0.09), 6.81) # I think this is what looked pretty good from part 1
        self.set_view(137, 41, 0, (0.48, 0.32, -0.52), 6.81)
        self.add(surfaces[0])
        self.add(grids[0])
        self.add(texts[0])
        self.wait()

        # self.set_view(135, 25, 0, (0.02, 0.1, -0.11), 7.56) #isometric, fiarly high
        # # self.wait()

        # self.set_view(137, 41, 0, (0.14, -0.04, -0.09), 6.81) #isometric, more down
        # self.wait()


        self.wait()
        pass





class PosterHackinExploreColormapsTwo(WelchScene):
    def construct(self):
    
        # for cmap in ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'gist_earth', 'terrain', 'turbo', 'hsv', 'Greens', 'Oranges', 'Reds', 'Blues', 'Purples', 'cool', 'nipy_spectral', 'seismic', 'rainbow', 'gist_rainbow']:
        for cmap in ['viridis_r', 'plasma_r', 'inferno_r', 'magma_r', 'jet_r', 'gist_earth_r', 'terrain_r', 'turbo_r', 'hsv_r', 'Greens_r', 'Oranges_r', 'Reds_r', 'Blues_r', 'Purples_r', 'cool_r', 'nipy_spectral_r', 'seismic_r', 'rainbow_r', 'gist_rainbow_r']:
            #Load up other surfaces to visualize
            loss_arrays=[]
            paths=[]
            texture_paths=[]
            print('Loading Surface Arrays...')

            render_paths=glob.glob(data_dir+'*')
            for r in render_paths:
            # r=render_paths[0]
                for np_path in glob.glob(r+'/*.npy'):
                    # print(np_path)
                    loss_arrays.append(np.load(np_path))
                    print(np_path, loss_arrays[-1].shape)
                    paths.append(np_path)
                    plt.clf()
                    plt.figure(frameon=False)
                    ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
                    ax.set_axis_off()
                    plt.gcf().add_axes(ax)
                    plt.imshow(np.rot90(loss_arrays[-1].T), cmap=cmap) #have to transpose if transposing u and v and param_surface_1
                    texture_paths.append(r+'/'+np_path.split('/')[-1].split('.')[0]+'_'+cmap+'_texture.png')
                    plt.savefig(texture_paths[-1], bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()

            self.wait()

            surfaces=m.Group()
            surf_functions=[] #Need this later to move dot around.
            grids=m.Group()
            texts=m.Group()
            print("Loading Surfaces and Gridlines...")
            for i in tqdm(range(len(paths[:MAX_RENDERS]))): #LIMIT HERE FOR TESTING
                if 'llama' in paths[i]: scaling=0.07
                elif 'gpt' in paths[i]: scaling=0.2
                elif 'qwen' in paths[i]: scaling=0.07
                elif 'gemma' in paths[i]: scaling=0.07
                else: scaling=0.07

                surf_func=partial(param_surface_3, surf_array=loss_arrays[i], scaling=scaling, surf_mean=np.mean(loss_arrays[i]))
                surf_functions.append(surf_func)
                surface = parametric_surface(
                    surf_func,  
                    u_range=[-2.5, 2.5], #Some have different ranges, but I don't think it matters?
                    v_range=[-2.5, 2.5],
                    resolution=(512, 512),
                )

                ts2 = textured_surface(surface, texture_paths[i])
                set_surface_shading(ts2, 0.0, 0.1, 0)
                surfaces.add(ts2)

                num_lines = 64  # Number of gridlines in each direction
                num_points = 512  # Number  of points per line
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

                t=m.Text(paths[i][-40:])
                t.scale(0.4)
                t.rotate(m.DEGREES*180)
                t.move_to([0, 3, 0])
                texts.add(t)

            # self.set_view(134, 30, 0, (0.12, 0.11, -0.15), 7.56)
            # self.set_view(132, 30, 0, (0.32, 0.4, -0.32), 6.82)
            self.set_view(132, 30, 0, (0.24, 0.27, -0.23), 7.50)
            self.add(surfaces[0])
            self.add(grids[0])
            self.add(texts[0])
            self.wait()

            for i in range(1, np.min([len(paths), MAX_RENDERS])):
                self.remove(surfaces[i-1])
                self.remove(grids[i-1])
                self.remove(texts[i-1])
                self.add(surfaces[i])
                self.add(grids[i])
                self.add(texts[i])
                self.wait()

            self.remove(surfaces[i])
            self.remove(grids[i])
            self.remove(texts[i])



        # self.set_view(135, 25, 0, (0.02, 0.1, -0.11), 7.56) #isometric, fiarly high
        # # self.wait()

        # self.set_view(137, 41, 0, (0.14, -0.04, -0.09), 6.81) #isometric, more down
        # self.wait()


        # self.wait(20)
        pass

