from manimlib import *
import numpy as np
import matplotlib.pyplot as plt

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b'
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

data_dir='/Users/stephen/Library/CloudStorage/Dropbox-Stephencwelch/welch_labs/resnet/hackin'
# data_dir='/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/resnet/hackin/'

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5

image_opacity=0.936  
still_hold=1.0
steps_per_viz=11     
fov=PI/3
kernel_k=0
block_cell=0.48

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)
# image_bounds=(-224.0, 224.0, -224.0, 224.0, 0.0, 3*pixel_dim)

#Camera keyframes: theta, phi, gamma, center, height
# p24_end=(-135.4168, 63.3448, 114.4748, (0.0, 0.0, 0.0), 96.224)
# p24d_end=(-139.5221, 31.6032, 134.9423, (0.0, 0.0, 0.0), 96.224)


viridis_lut=plt.get_cmap('viridis')(np.linspace(0, 1, 256))


def viridis(values):
    x=np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    idx=np.clip((x*256).astype(np.int32), 0, 255) #matplotlib bins with int(x*N), not round(x*(N-1))
    return viridis_lut[idx]

# 6 faces x 4 corners of a unit cube, wound counter-clockwise seen from outside
face_corners=np.array([
    [[-.5, -.5,  .5], [.5, -.5,  .5], [.5,  .5,  .5], [-.5,  .5,  .5]],
    [[ .5, -.5, -.5], [-.5, -.5, -.5], [-.5, .5, -.5], [ .5,  .5, -.5]],
    [[ .5, -.5,  .5], [.5, -.5, -.5], [.5,  .5, -.5], [ .5,  .5,  .5]],
    [[-.5, -.5, -.5], [-.5, -.5, .5], [-.5, .5,  .5], [-.5,  .5, -.5]],
    [[-.5,  .5,  .5], [.5,  .5,  .5], [.5,  .5, -.5], [-.5,  .5, -.5]],
    [[-.5, -.5, -.5], [.5, -.5, -.5], [.5, -.5,  .5], [-.5, -.5,  .5]],
], dtype=np.float64)

quad_tris=np.array([0, 1, 2, 0, 2, 3], dtype=int)
box_edges=[(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
           (0,4), (1,5), (2,6), (3,7)]


class VoxelBlock(Surface):
    """Every box in a layer batched into one mesh."""
    shader_folder='surface'
    render_primitive=moderngl.TRIANGLES

    def __init__(self, centers, sizes, rgba, view_forward=None, depth_test=None, **kwargs):
        centers=np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        rgba=np.asarray(rgba, dtype=np.float64).reshape(-1, 4)
        sizes=np.asarray(sizes, dtype=np.float64)
        if sizes.ndim==1:
            sizes=np.tile(sizes.reshape(1, 3), (len(centers), 1))

        # Painter's algorithm: with the depth test on, the nearest translucent cell writes
        # depth and hides the rest, flattening a 64 layer block to one face
        if view_forward is not None and len(centers):
            f=np.asarray(view_forward, dtype=np.float64)
            order=np.argsort(-(centers@(f/np.linalg.norm(f)))) #farthest first
            centers, sizes, rgba=centers[order], sizes[order], rgba[order]
        if depth_test is None:
            depth_test=view_forward is None

        self.centers=centers
        self.sizes=sizes
        self.rgba=rgba
        super().__init__(shading=(0, 0, 0), depth_test=depth_test, **kwargs) #Faces read flat

    @Surface.affects_data
    def init_points(self):
        n=len(self.centers)
        if n==0:
            self.set_points(np.zeros((0, 3)))
            self.tri_indices=np.zeros(0, dtype=int)
            return

        corners=face_corners[None,:,:,:]*self.sizes[:,None,None,:]+self.centers[:,None,None,:]
        verts=corners.reshape(-1, 3)
        self.set_points(verts)

        # The shader takes normals from (du_point - point) x (dv_point - point)
        c0=corners[:,:,0,:]
        du=np.repeat((corners[:,:,1,:]-c0)[:,:,None,:], 4, axis=2)
        dv=np.repeat((corners[:,:,3,:]-c0)[:,:,None,:], 4, axis=2)
        self.data['du_point'][:]=verts+du.reshape(-1, 3)
        self.data['dv_point'][:]=verts+dv.reshape(-1, 3)

        self.tri_indices=(np.arange(n*6)*4).repeat(6)+np.tile(quad_tris, n*6)

    def compute_triangle_indices(self):
        self.triangle_indices=self.tri_indices
        return self.triangle_indices

    def init_colors(self):
        if len(self.centers):
            self.data['rgba'][:]=np.repeat(self.rgba, 24, axis=0)


def conv_data_block(a, start_depth, vmin=None, vmax=None, keep=None, cell_size=1.0,
                    alpha=0.5, view_forward=None):
    a=np.asarray(a, dtype=np.float64)
    n_c, n_i, n_j=a.shape
    kk, ii, jj=np.meshgrid(np.arange(n_c), np.arange(n_i), np.arange(n_j), indexing='ij')

    vmin=a.min() if vmin is None else vmin
    vmax=a.max() if vmax is None else vmax
    vals=(a-vmin)/(vmax-vmin)          #imshow's default min-max stretch
    keep=np.ones(a.shape, dtype=bool) if keep is None else keep

    half=np.floor(n_j/2)
    centers=np.stack([(jj[keep]-half)*cell_size, (-ii[keep]+half)*cell_size,
                      depth_step*kk[keep]+start_depth], axis=-1)
    rgba=viridis(vals[keep])
    rgba[:,3]=alpha

    # block=VoxelBlock(centers, np.array([cell_size, cell_size, cell_depth]), rgba)
    block=VoxelBlock(centers, np.array([cell_size, cell_size, cell_depth]), rgba,
                     view_forward=view_forward)
    half_extent=(half+0.5)*cell_size
    bounds=(-half_extent, half_extent, -half_extent, half_extent,
            start_depth, n_c*depth_step+start_depth)
    return block, bounds

def reveal_mask(shape, i, j, k):
    """Raster-order reveal up to cell (i, j), channels <= k."""
    n_c, n_i, n_j=shape
    revealed=(np.arange(n_i)[:,None]*n_j+np.arange(n_j)[None,:])<=(i*n_j+j)
    mask=np.zeros(shape, dtype=bool)
    mask[:k+1]=revealed
    return mask


def image_plane(im_path, opacity):
    img=ImageMobject(im_path)
    img.set_width(image_bounds[1]-image_bounds[0], stretch=True)
    img.set_height(image_bounds[3]-image_bounds[2], stretch=True)
    img.set_opacity(opacity)
    img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
    return img


def kernel_weights_block(weights, extent):
    """A conv-1 filter painted into its patch, black through to magenta."""
    w=np.asarray(weights, dtype=np.float64)
    w=w-w.min()
    min_x, max_x, _, max_y=extent
    step=(max_x-min_x)/w.shape[1]

    kk, ii, jj=np.meshgrid(np.arange(w.shape[0]), np.arange(w.shape[1]), np.arange(w.shape[2]),
                           indexing='ij')
    vals=(w/w.max()).ravel()
    centers=np.stack([jj.ravel()*step+min_x, -ii.ravel()*step+max_y,
                      kk.ravel()*pixel_dim], axis=-1)
    rgba=np.zeros((len(vals), 4))
    rgba[:,0]=vals
    rgba[:,2]=vals
    rgba[:,3]=0.5
    return VoxelBlock(centers, np.array([1.0, 1.0, depth_step]), rgba)


def polyline(points, color, radius):
    """One independent segment per step, since manim miters the joins of a continuous path."""
    pts=np.asarray(points, dtype=np.float64)
    group=VGroup()
    for start, end in zip(pts[:-1], pts[1:]):
        if np.allclose(start, end):
            continue
        seg=Line(start, end)
        seg.set_stroke(color, width=200*radius, opacity=1.0) #Width in world units
        seg.set_fill(opacity=0.0)
        seg.set_scale_stroke_with_zoom(True)
        seg.apply_depth_test()
        group.add(seg)
    return group


def prism(min_x, max_x, min_y, max_y, min_z, max_z, color, radius):
    vertices=np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z],
        [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z],
        [max_x, max_y, max_z], [min_x, max_y, max_z],
    ], dtype=np.float64)
    group=VGroup()
    for p, q in box_edges:
        group.add(*polyline([vertices[p], vertices[q]], color, radius))
    return group


def conv1_kernel(i, j, k, act_shape, weights, cell_size=1.0, stride=1):
    """The conv-1 kernel wired to its activation, with the filter painted into the patch."""
    min_x, max_x, min_y, _, min_z, max_z=image_bounds
    group=Group()

    #Activation cell (i, j) -> its footprint on the image plane
    step=(max_x-min_x)/act_shape[-1]      #image units per activation cell
    px=step/stride                        #image units per input pixel
    size=weights.shape[-1]*px             #receptive field of the kernel

    half=np.floor(act_shape[-1]/2)
    hc=0.5*cell_size
    dst_x=(j-half)*cell_size
    dst_y=(-i+half)*cell_size
    dst_z=depth_step*k+spacing_between_layers+1

    src_x=min_x+step*j
    src_y=-min_y-step*i  #-min_y spells max_y; the bounds are symmetric

    connectors=[
        [(dst_x-hc, dst_y-hc, dst_z), (src_x,      src_y-size, max_z)],
        [(dst_x+hc, dst_y-hc, dst_z), (src_x+size, src_y-size, max_z)],
        [(dst_x-hc, dst_y+hc, dst_z), (src_x,      src_y,      max_z)],
        [(dst_x+hc, dst_y+hc, dst_z), (src_x+size, src_y,      max_z)],
    ]
    for cc in connectors:
        group.add(polyline(cc, MAGENTA, line_radius))
    for p, q in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        group.add(polyline([connectors[p][0], connectors[q][0]], MAGENTA, line_radius))

    extent=(src_x, src_x+size, src_y-size, src_y)
    group.add(prism(extent[0], extent[1], extent[2], extent[3], min_z, max_z, MAGENTA,
                    line_radius))
    group.add(kernel_weights_block(weights, extent))
    return group


def masked_conv1(a, i, j, k):
    """Conv-1 revealed in raster order up to cell (i, j) of channel <= k."""
    out=np.zeros_like(a)
    n_j=a.shape[2]
    revealed=(np.arange(a.shape[1])[:,None]*n_j+np.arange(n_j)[None,:])<=(i*n_j+j)
    #The floor keeps near-zero revealed cells above the render threshold
    out[:k+1]=np.where(revealed, np.maximum(a[:k+1], 0.01*a[0].max()), 0.0)
    return out


def swap_out(scene, mobject):
    """Scene.remove leaves id_to_mobject_map, reference cycles and moderngl buffers behind,
    so without this a per-frame shot leaks until the machine runs out of memory."""
    if mobject is None:
        return
    scene.remove(mobject)
    for sm in mobject.get_family():
        scene.id_to_mobject_map.pop(id(sm), None)
        if sm.shader_wrapper is not None:
            sm.shader_wrapper.release() #Buffers only: the textures are shared, ctx-wide caches
        sm.family=None
        sm.parents.clear()
        sm.submobjects.clear()
        sm.data=None
        sm.shader_wrapper=None
        for attr in ('centers', 'sizes', 'rgba', 'tri_indices', 'triangle_indices'):
            if hasattr(sm, attr):
                setattr(sm, attr, None)


def clear_scene(scene):
    for mob in list(scene.mobjects):
        if mob is not scene.camera.frame:
            swap_out(scene, mob)
    scene.clear()
    scene.id_to_mobject_map.clear()
    scene.add(scene.camera.frame)


def blend_views(a, b, t):
    return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t,
            tuple(p+(q-p)*t for p, q in zip(a[3], b[3])), a[4]+(b[4]-a[4])*t)


def orient(mob):
    mob.rotate(90*DEGREES, [0, 1, 0], about_point=ORIGIN)
    mob.rotate(90*DEGREES, [1, 0, 0], about_point=ORIGIN)
    return mob

def activation_image_grid(image_dir, n_c, start_depth, map_width, pitch,
                          skip_channel=None, n_cols=8):
    group=Group()
    for c in range(n_c):
        if c==skip_channel:
            continue
        r, col=divmod(c, n_cols)
        img=ImageMobject(f'{image_dir}/act_{c:02d}.png')
        img.set_width(map_width, stretch=True)
        img.set_height(map_width, stretch=True)
        img.move_to([col*pitch-0.5*block_cell, -r*pitch+0.5*block_cell, start_depth])
        group.add(img)
    return group

def make_channel_images(im_path, out_dir):
    """Split the input into R/G/B-only PNGs, once."""
    from PIL import Image
    import os
    paths=[]
    src=None
    for ch, name in enumerate('rgb'):
        p=f'{out_dir}/lemon_{name}.png'
        paths.append(p)
        if not os.path.exists(p):
            if src is None:
                src=np.array(Image.open(im_path).convert('RGB'))
            solo=np.zeros_like(src)
            solo[...,ch]=src[...,ch]
            Image.fromarray(solo).save(p)
    return paths

class P13(InteractiveScene):
    def construct(self):
        act=np.load(data_dir+'/p13/lemon_activations_47587.npy', allow_pickle=True).item()
        layer_1_weights=np.load(data_dir+'/p13/plain_8_conv_1.npy')

        start_position=(15, 52, 0, (np.float32(4.48), np.float32(4.88), np.float32(-6.58)), 106.23)
        end_position=(61, 73, 0, (np.float32(3.79), np.float32(7.16), np.float32(-1.82)), 94.39)


        thresh=-100
        a=act['conv1'][0]
        temp=a[0].copy()
        a[0]=a[22] #Start with nice vertical edges
        a[22]=temp

        #Static geometry
        image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))
        img=orient(image_plane(data_dir+'/p13/lemon.jpg', opacity=0.6))

        n_i, n_j=a.shape[1], a.shape[2]

        vmin=float(a[kernel_k].min())
        vmax=float(a[kernel_k].max())

        #bounds (static border)
        _, bounds=conv_data_block(a, spacing_between_layers+1, vmin=vmin, vmax=vmax,
                                  cell_size=block_cell, alpha=0.6)


        conv_1_border=orient(prism(*bounds, CHILL_BROWN, line_radius))

        self.add(img, image_border, conv_1_border)
        self.frame.reorient(32, 66, 0, (np.float32(6.76), np.float32(11.09), np.float32(-0.32)), 106.23)
        self.wait(1)

        quick_mode=True   #flip to False for the real render

        if quick_mode:
            block, _=conv_data_block(a, spacing_between_layers+1, vmin=vmin, vmax=vmax,
                                     keep=reveal_mask(a.shape, n_i-1, n_j-1, kernel_k),
                                     cell_size=block_cell)
            orient(block)
            self.add(block)
            self.frame.reorient(*end_position)
            self.wait(0.1)
        else:
            block=None
            kernel=None
            positions=list(np.ndindex(n_i, n_j))
            n_steps=len(positions)
            for step, (i, j) in enumerate(positions):
                last=(step==n_steps-1)
                if step%steps_per_viz!=0 and not last:
                    continue

                swap_out(self, kernel)
                swap_out(self, block)

                kernel=orient(conv1_kernel(i, j, kernel_k, a.shape, layer_1_weights[0],
                                           cell_size=block_cell, stride=1))
                block, _=conv_data_block(a, spacing_between_layers+1, vmin=vmin, vmax=vmax,
                                         keep=reveal_mask(a.shape, i, j, kernel_k),
                                         cell_size=block_cell)
                orient(block)
                self.add(block, kernel)

                t=smooth(step/(n_steps-1))   #ease in/out over the whole sweep
                self.frame.reorient(*blend_views(start_position, end_position, t))
                self.wait(1/30)
            swap_out(self, kernel)
        self.wait(still_hold)


        flat_view=(90, 90, 0, (np.float32(32.08), np.float32(202.27), np.float32(-200.76)), 456.79)
        # self.frame.reorient(90, 90, 0, (np.float32(32.15), np.float32(192.87), np.float32(-191.94)), 479.34)
        pitch=n_j*block_cell+4.0     #map width ~26.9 + gap

        #Fade everything but the map; bring the map to full opacity
        self.play(FadeOut(img), FadeOut(image_border), FadeOut(conv_1_border),
                  block.animate.set_opacity(1.0), run_time=2.0)
        # swap_out(self, img); 
        # swap_out(self, image_border); 
        # swap_out(self, conv_1_border)
        # img=image_border=conv_1_border=None

        #The other 63 maps; channel 0 (your swapped-in vertical edge map) stays put as upper-left
        grid=activation_image_grid(data_dir+'/p13/conv_1_activations', a.shape[0],
                                   spacing_between_layers+1,
                                   n_j*block_cell, pitch, skip_channel=kernel_k)
        orient(grid)
        grid.set_opacity(0.0)
        self.add(grid)
        self.play(grid.animate.set_opacity(1.0),
                  self.frame.animate.reorient(*flat_view), run_time=12.0)

        self.wait()

        ## P15 ---- Stack the maps back into a tensor ----
        n_c=a.shape[0]
        z0=spacing_between_layers+1
        spread_step=10*depth_step   #spread stack fills a prism 6x the depth of conv_1_border
        wide_bounds=(bounds[0], bounds[1], bounds[2], bounds[3],
                     z0, z0+n_c*spread_step)

        def orient_point(p):
            x, y, z=p
            return np.array([z, x, y])  #the same permutation orient() applies about ORIGIN

        stack_x, stack_y=-0.5*block_cell, 0.5*block_cell  #channel-0's slot in the grid

        wide_border=orient(prism(*wide_bounds, CHILL_BROWN, line_radius))
        spread_view=(38, 63, 0, (np.float32(37.84), np.float32(7.36), np.float32(-11.49)), 138.34)

        channels=[c for c in range(n_c) if c!=kernel_k]
        gather=[m.animate.move_to(orient_point([stack_x, stack_y, z0+c*spread_step]))
                for m, c in zip(grid, channels)]

        self.wait()
        self.play(LaggedStart(*gather, lag_ratio=0.01),
                  FadeIn(wide_border),
                  FadeIn(img), 
                  FadeIn(image_border),
                  self.frame.animate.reorient(*spread_view),
                  run_time=8.0)
        self.wait(still_hold)

        # Ok now add dimensions in premiere
        # Then: 
        # "These activations form a new sort of image, but where our input image has 
        # red green and blue color channels, our activation tensor has 64 channels, 
        # each corresponding to a different type of image feature."
        #
        # Ok so maybe dope idea here, temporarily fade out all activations, 
        # Flip image through red, green, and blue color channels, and then 
        # quickly roll through adding back each activation image. 

        img=orient(image_plane(data_dir+'/p13/lemon.jpg', opacity=0.4))
        image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))
        interlude_view=(38, 68, 0, (10.0, 8.0, -2.0), 120)  #placeholder; tune in embed

        self.wait(1)
        self.remove(grid)
        self.remove(block)
        self.wait(1)
        # grid.set_opacity(0.04)
        # block.set_opacity(0.04)
        # self.play(grid.animate.set_opacity(0.01), block.animate.set_opacity(0.01), run_time=2.0)  #wide_border stays


        # self.play(FadeIn(img), FadeIn(image_border),
        #           self.frame.animate.reorient(*interlude_view), run_time=3.0)
        # self.wait(still_hold)

        ## ---- Flip through R, G, B, then back to color ----
        channel_paths=make_channel_images(data_dir+'/p13/lemon.jpg', data_dir+'/p13')
        current=img

        self.wait()
        for p in channel_paths:
            nxt=orient(image_plane(p, opacity=1.0))
            self.remove(current)
            self.add(nxt)
            # self.play(FadeIn(nxt), FadeOut(current), run_time=1.0)
            swap_out(self, current)
            current=nxt
            self.wait(0.5)
        img=orient(image_plane(data_dir+'/p13/lemon.jpg', opacity=0.4))
        # self.play(FadeIn(img), FadeOut(current), run_time=1.0)
        self.wait()
        self.remove(current)
        self.add(img)
        swap_out(self, current)
        

        ## ---- Bring the activation maps back one at a time ----
        block.set_opacity(1.0)  
        grid.set_opacity(1.0)

        cascade_start=interlude_view  #or self.frame.get_... wherever you are at this point
        cascade_end=(58, 69, 0, (np.float32(40.75), np.float32(13.26), np.float32(-9.89)), 138.34)

        click_hold=0.15   #seconds per map; 65 maps ≈ 10s total
        members=[block, *grid]
        n=len(members)

        self.wait(still_hold)
        for idx, m in enumerate(members):
            self.add(m)
            t=smooth(idx/(n-1))
            # self.frame.reorient(*blend_views(cascade_start, cascade_end, t))
            self.wait(click_hold)
        self.wait(still_hold)

        #Optionally send the image away again before the compress:
        self.play(FadeOut(img), FadeOut(image_border), run_time=1.5)
        swap_out(self, img); swap_out(self, image_border)
        img=image_border=None







        ## ---- Compress into the tensor, return to end_position ----
        conv_1_border=orient(prism(*bounds, CHILL_BROWN, line_radius)) #original was swapped out
        squeeze=[m.animate.move_to(orient_point([stack_x, stack_y, z0+c*depth_step]))
                 for m, c in zip(grid, channels)]

        self.play(LaggedStart(*squeeze, lag_ratio=0.005),
                  FadeOut(wide_border), FadeIn(conv_1_border),
                  self.frame.animate.reorient(*end_position),
                  run_time=6.0)
        swap_out(self, wide_border)
        self.wait(still_hold)

        ## ---- Optional: crossfade the card stack into a real voxel block ----
        a_norm=(a-a.min(axis=(1,2), keepdims=True))/np.ptp(a, axis=(1,2), keepdims=True)
        cam=self.frame.get_implied_camera_location()
        fwd_world=self.frame.get_center()-cam
        fwd=np.array([fwd_world[1], fwd_world[2], fwd_world[0]]) #undo orient's permutation
        full_block, _=conv_data_block(a_norm, z0, vmin=0.0, vmax=1.0,
                                      cell_size=block_cell, alpha=0.6, view_forward=fwd)
        orient(full_block)
        self.play(FadeIn(full_block), FadeOut(grid), FadeOut(block), run_time=2.0)
        swap_out(self, grid); swap_out(self, block)
        block=full_block



        self.wait(20)
        self.embed()





        # #Sweep
        # block=None
        # kernel=None
        # positions=list(np.ndindex(n_i, n_j))
        # n_steps=len(positions)
        # for step, (i, j) in enumerate(positions):
        #     last=(step==n_steps-1)
        #     if step%steps_per_viz!=0 and not last:
        #         continue

        #     swap_out(self, kernel)
        #     swap_out(self, block)

        #     kernel=orient(conv1_kernel(i, j, kernel_k, a.shape, layer_1_weights[0],
        #                                cell_size=block_cell, stride=1))
        #     block, _=conv_data_block(masked_conv1(a, i, j, kernel_k),
        #                              spacing_between_layers+1, 0.005, vmax=vmax,
        #                              cell_size=block_cell)
        #     orient(block)
        #     self.add(block, kernel)

        #     t=smooth(step/(n_steps-1))   #ease in/out over the whole sweep
        #     self.frame.reorient(*blend_views(start_position, end_position, t))
        #     self.wait(1/30)

        # swap_out(self, kernel)       #drop the kernel at the end, leave the filled map
        # self.wait(still_hold)
        



# class P13(InteractiveScene):
#     def construct(self):

        
#         act=np.load(data_dir+'/p13/lemon_activations_47587.npy', allow_pickle=True).item()
#         layer_1_weights=np.load(data_dir+'/p13/plain_8_conv_1.npy')


#         ##Ok lets start simple with just the image here.
#         image_border=prism(*image_bounds, CHILL_BROWN, line_radius)
#         img=image_plane(data_dir+'/p13/lemon.jpg', opacity=0.4)


#         kernel_k=0 #20 is nice vertical edges I think, maybe just transpose 1 and 20

#         a=act['conv1'][0]

#         block, bounds=conv_data_block(masked_conv1(a, 0, 10, kernel_k),
#                                       spacing_between_layers+1, 0.005, cell_size=0.48)
#         conv_1_border=prism(*bounds, CHILL_BROWN, line_radius)

#         cell=(image_bounds[1]-image_bounds[0])/a.shape[-1]
#         k=conv1_kernel(0, 10, kernel_k, a.shape, layer_1_weights[0], cell_size=0.48, stride=1)


#         # self.add(conv1_kernel(0, 10, kernel_k, a.shape, weights, forward))
#         net_group=Group(img, image_border, conv_1_border, k)
#         net_group.rotate(90*DEGREES, [0, 1, 0])
#         net_group.rotate(90*DEGREES, [1, 0, 0])
#         self.add(net_group)

#         # image_border.move_to([0, 10, 0])

#         self.frame.reorient(32, 66, 0, (np.float32(6.76), np.float32(11.09), np.float32(-0.32)), 106.23)


#         self.wait()





        # self.camera.background_rgba=[0, 0, 0, 1]
        # self.frame.set_field_of_view(fov)
        # a=load_activation('hot_dog', 'features_2')
        # weights=load_activation('weights', 'features0')[0]
        # vmax=float(a[0].max())

        # self.frame.reorient(*p24_end)
        # forward=forward_from(p24_end)
        # self.add(prism(*image_bounds, WHITE, line_radius))
        # self.add(conv1_kernel(0, 10, kernel_k, a.shape, weights, forward))
        # block, bounds=conv_data_block(masked_conv1(a, 0, 10, kernel_k),
        #                               spacing_between_layers+1, 0.005, forward)
        # self.add(block)
        # self.add(prism(*bounds, WHITE, line_radius))
        # self.wait(still_hold)


        self.wait(20)
        self.embed()





