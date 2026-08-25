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

data_dir='/Volumes/hot_1/Stephencwelch Dropbox/welch_labs/resnet/hackin/'
sweep_dir=data_dir+'p26_sweep_2/'

spacing_between_layers=5
line_radius=0.18
depth_step=0.125
cell_depth=0.1
pixel_dim=0.5
block_cell=0.48
still_hold=1.0

image_bounds=(-32.0, 32.0, -32.0, 32.0, 0.0, 3*pixel_dim)

#Layer depths, carried over from p13_30: layer 1 compressed to 64*squish_step=20 world units
z0=spacing_between_layers+1
base_depth=64*(10*depth_step/4)              #20
depth_mults={64: 1.0, 128: 1.35, 256: 1.8}
deep_keys=['relu', 'layer1.0.relu', 'layer1.0', 'layer2.0.relu', 'layer2.0',
           'layer3.0.relu', 'layer3.0']       #the 7 post-ReLU tensors of plain8

#fc column and probability plot
fc_cell=1.35                                  #was pooled_cell=14*0.48/5 in p13_30
fc_factor=1.2                                 #fc height / layer3.0 depth
fc_gap=4.0                                    #fc column -> index axis; one-sided plot, so this can be small
plot_w=11.0                                   #world units for prob_axis_max
prob_axis_max=1.0                             #fixed so sweep frames are comparable; drop toward max prob to zoom


viridis_lut=plt.get_cmap('viridis')(np.linspace(0, 1, 256))


def viridis(values):
    x=np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    idx=np.clip((x*256).astype(np.int32), 0, 255) #matplotlib bins with int(x*N), not round(x*(N-1))
    return viridis_lut[idx]


def softmax(x):
    e=np.exp(x-x.max())
    return e/e.sum()


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
                    alpha=0.5, z_step=depth_step, view_forward=None):

    a=np.asarray(a, dtype=np.float64)
    n_c, n_i, n_j=a.shape
    kk, ii, jj=np.meshgrid(np.arange(n_c), np.arange(n_i), np.arange(n_j), indexing='ij')

    vmin=a.min() if vmin is None else vmin
    vmax=a.max() if vmax is None else vmax
    vals=(a-vmin)/(vmax-vmin)          #imshow's default min-max stretch
    keep=np.ones(a.shape, dtype=bool) if keep is None else keep

    half=np.floor(n_j/2)
    centers=np.stack([(jj[keep]-half)*cell_size, (-ii[keep]+half)*cell_size,
                      z_step*kk[keep]+start_depth], axis=-1)
    rgba=viridis(vals[keep])
    rgba[:,3]=alpha

    block=VoxelBlock(centers, np.array([cell_size, cell_size, cell_depth]), rgba,
                     view_forward=view_forward)
    half_extent=(half+0.5)*cell_size
    bounds=(-half_extent, half_extent, -half_extent, half_extent,
            start_depth, n_c*z_step+start_depth)
    return block, bounds


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


def image_plane(im_path, opacity):
    img=ImageMobject(im_path)
    img.set_width(image_bounds[1]-image_bounds[0], stretch=True)
    img.set_height(image_bounds[3]-image_bounds[2], stretch=True)
    img.set_opacity(opacity)
    img.move_to([0, 0, 0.5*(image_bounds[4]+image_bounds[5])])
    return img


def orient(mob):
    """Network depth z -> world x, image x -> world y, image y -> world z."""
    mob.rotate(90*DEGREES, [0, 1, 0], about_point=ORIGIN)
    mob.rotate(90*DEGREES, [1, 0, 0], about_point=ORIGIN)
    return mob


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


def layer_stats(r, pct=97):
    """Per-channel max and per-channel percentile threshold (of the normalized map)."""
    r=np.asarray(r, dtype=np.float64)
    vmax=r.max(axis=(1, 2), keepdims=True)
    vmax[vmax==0]=1.0
    thresh=np.percentile(r/vmax, pct, axis=(1, 2), keepdims=True)
    return vmax, thresh


def relu_viz_block(r, z0, z_step, cell, pct=97, alpha=0.7, stats=None):
    """Per-channel normalized, per-channel percentile-thresholded, one merged block.
    Pass `stats` from a reference frame to hold the normalization fixed across a sweep;
    otherwise every frame renormalizes and always shows exactly (100-pct)% of each channel."""
    r=np.asarray(r, dtype=np.float64)
    vmax, thresh=layer_stats(r, pct) if stats is None else stats
    rn=r/vmax
    return conv_data_block(rn, z0, vmin=0.0, vmax=1.0, keep=(rn>thresh),
                           cell_size=cell, alpha=alpha, z_step=z_step)


def cap_with_triangle(axis, at_start=False, length=2.5, width=2.0, normal=UP, color=CHILL_BROWN):
    """Filled triangle on one end of a NumberLine, in the plane perpendicular to `normal`.
    Shortens the line to the triangle's base so the two don't z-fight."""
    p0, p1=axis.get_start(), axis.get_end()
    tip, d=(p0, p0-p1) if at_start else (p1, p1-p0)
    d=d/np.linalg.norm(d)
    perp=np.cross(normal, d)
    base=tip-length*d
    tri=Polygon(tip, base+0.5*width*perp, base-0.5*width*perp)
    tri.set_fill(color, 1.0)
    tri.set_stroke(width=0)
    tri.apply_depth_test()
    if at_start:
        axis.put_start_and_end_on(base, p1)
    else:
        axis.put_start_and_end_on(p0, base)
    return tri


## ---- Sweep helpers: geometry is fixed by the shapes, only the values change per file ----

def load_act(sweep_id):
    return np.load(f'{sweep_dir}{sweep_id:03d}.npy', allow_pickle=True).item()


def layer_layout(act):
    """(key, z_start, z_step) for each conv layer, plus the un-oriented depth of the fc column."""
    layout=[]
    z=z0
    for key in deep_keys:
        n_c=act[key].shape[1]
        depth=base_depth*depth_mults[n_c]
        layout.append((key, z, depth/n_c))
        z+=depth+spacing_between_layers
    return layout, z


def fc_geometry(act, fc_z):
    n_fc=act['fc'].shape[-1]
    fc_height=fc_factor*base_depth*depth_mults[256]     #relative to layer3.0 depth, as in p13_30
    fc_step=fc_height/n_fc
    return n_fc, fc_height, fc_step


def fc_slot_centers(n_fc, fc_step, fc_z):
    idx=np.arange(n_fc)
    y=(0.5*(n_fc-1)-idx)*fc_step       #index 0 at top
    return np.stack([np.zeros(n_fc), y, np.full(n_fc, fc_z)], axis=-1)


def build_blocks(act, layout, stats=None):
    """The 7 thresholded conv blocks for one activation file, oriented."""
    blocks=[]
    for li, (key, z, z_step) in enumerate(layout):
        s=None if stats is None else stats[li]
        blk, _=relu_viz_block(act[key][0], z, z_step, block_cell, stats=s)
        blocks.append(orient(blk))
    return blocks


def build_fc(probs, fc_step, fc_z, color_max):
    """Vertical column of 1000 wafers colored by softmax probability."""
    n_fc=len(probs)
    rgba=viridis(probs/color_max)
    rgba[:,3]=0.7
    wafer=np.array([fc_cell, cell_depth, fc_cell])   #thin slice per unit, stacked in y
    return orient(VoxelBlock(fc_slot_centers(n_fc, fc_step, fc_z), wafer, rgba))


def build_prob_axes(n_fc, fc_step, fc_z):
    """Axes with index pointing down and a one-sided probability axis pointing right.
    Returns (axes, tips, prob_unit); c2p(0, 0) sits at the top of the index axis."""
    prob_unit=plot_w/prob_axis_max
    y_max=1.05*n_fc

    axes=Axes(x_range=(0, prob_axis_max, prob_axis_max), y_range=(0, y_max, y_max),
              width=prob_axis_max*prob_unit, height=y_max*fc_step,
              axis_config=dict(stroke_width=8, include_ticks=False, include_tip=False))
    axes.set_color(CHILL_BROWN)
    for axis in axes.get_axes():
        axis.set_scale_stroke_with_zoom(True)
        axis.apply_depth_test()

    axes.rotate(-90*DEGREES, RIGHT, about_point=ORIGIN)   #+y -> -z: index 0 at the top
    axis_x=fc_z+fc_gap
    top_z=0.5*(n_fc-1)*fc_step
    axes.shift(np.array([axis_x, 0, top_z])-axes.c2p(0, 0))

    o=axes.c2p(0, 0)
    axes.stretch(1.02, 2, about_point=o)   #dim 2 = world z after the rotate; lines the plot up with the wafers

    tips=VGroup(cap_with_triangle(axes.x_axis, length=1.0, width=0.85),   #positive end of the prob axis
                cap_with_triangle(axes.y_axis, length=1.0, width=0.85))   #bottom of the index axis
    return axes, tips, prob_unit


def build_prob_curve(axes, probs, prob_unit, color_max):
    n_fc=len(probs)
    curve=VMobject()
    curve.set_points_as_corners(axes.c2p(probs, np.arange(n_fc)))
    curve.set_stroke(width=6, opacity=1.0)
    curve.set_scale_stroke_with_zoom(True)
    curve.apply_depth_test()
    curve.set_joint_type('bevel')

    #Same viridis map as the fc column, read back off the point positions
    o=axes.c2p(0, 0)
    pts=curve.get_points()
    v=(pts[:,0]-o[0])/prob_unit
    rgba=viridis(v/color_max)
    rgba[:,3]=1.0
    curve.data['stroke_rgba'][:]=rgba
    return curve


class P25_25(InteractiveScene):
    def construct(self):
        start_id=63
        sweep_ids=list(range(0, 129, 2))     #0 -> 128; thin this out or reorder as you like
        sweep_hold=0.25                       #seconds per sweep frame
        fixed_norm=True                       #normalize/threshold every frame with start_id's stats
        side_view=(0, 90, 0, (115.0, 0.0, 0.0), 150.0)   #tune in embed

        act=load_act(start_id)
        layout, fc_z=layer_layout(act)
        n_fc, fc_height, fc_step=fc_geometry(act, fc_z)

        #Stats from the starting frame, reused for every sweep frame when fixed_norm
        stats=[layer_stats(act[key][0]) for key, _, _ in layout] if fixed_norm else None

        def color_max(probs):
            return prob_axis_max if fixed_norm else float(probs.max())

        ## ---- Static geometry: image, one border per layer, fc border, axes ----
        img=orient(image_plane(data_dir+'/p25/screwdriver.jpg', opacity=0.6))   #edge-on from a pure side view
        image_border=orient(prism(*image_bounds, CHILL_BROWN, line_radius))

        borders=[]
        for key, z, z_step in layout:
            _, bnds=conv_data_block(act[key][0], z, cell_size=block_cell, z_step=z_step)
            borders.append(orient(prism(*bnds, CHILL_BROWN, line_radius)))

        hp=0.5*fc_cell
        hh=0.5*fc_height
        fc_border=orient(prism(-hp, hp, -hh, hh, fc_z-hp, fc_z+hp, CHILL_BROWN, line_radius))

        axes, tips, prob_unit=build_prob_axes(n_fc, fc_step, fc_z)

        ## ---- First frame ----
        probs=softmax(act['fc'][0])
        blocks=build_blocks(act, layout, stats)
        fc_block=build_fc(probs, fc_step, fc_z, color_max(probs))
        curve=build_prob_curve(axes, probs, prob_unit, color_max(probs))

        self.frame.reorient(*side_view)
        self.add(img, image_border)
        self.wait(1)

        fades=[AnimationGroup(FadeIn(b), FadeIn(p)) for b, p in zip(blocks, borders)]
        fades.append(AnimationGroup(FadeIn(fc_block), FadeIn(fc_border)))
        self.play(LaggedStart(*fades, lag_ratio=0.2), run_time=6.0)
        self.wait(still_hold)

        # self.play(ShowCreation(axes), FadeIn(tips), run_time=2.0)
        # self.play(ShowCreation(curve), run_time=3.0)
        # self.wait(still_hold)

        ## ---- Sweep: rebuild only what depends on the file ----
        net=Group(*blocks, fc_block, curve)
        for sid in sweep_ids:
            act=load_act(sid)
            probs=softmax(act['fc'][0])
            new=Group(*build_blocks(act, layout, stats),
                      build_fc(act['fc'][0], fc_step, fc_z, color_max(act['fc'][0])),
                      # build_prob_curve(axes, probs, prob_unit, color_max(probs))
                      )
            swap_out(self, net)
            net=new
            self.add(net)
            self.wait(sweep_hold)
        self.wait(still_hold)














        self.wait(20)
        self.embed()


        # Plain8
        # conv1 (1, 64, 112, 112)
        # bn1 (1, 64, 112, 112)
        # relu (1, 64, 112, 112)
        # maxpool (1, 64, 56, 56)
        # layer1.0.conv1 (1, 64, 56, 56)
        # layer1.0.bn1 (1, 64, 56, 56)
        # layer1.0.relu (1, 64, 56, 56)
        # layer1.0.conv2 (1, 64, 56, 56)
        # layer1.0.bn2 (1, 64, 56, 56)
        # layer1.0 (1, 64, 56, 56)
        # layer2.0.conv1 (1, 128, 28, 28)
        # layer2.0.bn1 (1, 128, 28, 28)
        # layer2.0.relu (1, 128, 28, 28)
        # layer2.0.conv2 (1, 128, 28, 28)
        # layer2.0.bn2 (1, 128, 28, 28)
        # layer2.0 (1, 128, 28, 28)
        # layer3.0.conv1 (1, 256, 14, 14)
        # layer3.0.bn1 (1, 256, 14, 14)
        # layer3.0.relu (1, 256, 14, 14)
        # layer3.0.conv2 (1, 256, 14, 14)
        # layer3.0.bn2 (1, 256, 14, 14)
        # layer3.0 (1, 256, 14, 14)
        # avgpool (1, 256, 1, 1)
        # fc (1, 1000)
        # image (224, 224, 3)