from welch_paths import welch_path
import manimgx as gx
from kt_colors import KT_AQUA, KT_BLUE, KT_GREEN, KT_LT_BLUE, KT_ORANGE, KT_PINK, KT_PURPLE, KT_RED, KT_YELLOW
from welch_style import BLUE, CHILL_BLUE, CHILL_BROWN, CHILL_GREEN, CYAN, FRESH_TAN, GREEN, MAGENTA, YELLOW, YELLOW_FADE
from resnet_assets import ACTIVATION_ROOT
from resnet_stream import GlideTo, SpinTo, SquashTo, block_layout, channel_pool, front_square, key_input_png, load_meta, load_stream, max_grid_png, stream_tensors, to_front_plane, upsampled_png
from welch_scene import WelchScene
from resnet_geometry import VoxelBlock, conv_data_block, layer_stats, orient, polyline, prism, viridis
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
from pathlib import Path


data_dir = str(ACTIVATION_ROOT) + '/'
svg_dir=welch_path('resnet/graphics/p63_to_manim')
cache_dir=data_dir+'p63_cache/'                      #from p63_cache_activations.ipynb
key_image_name='0111.png'                            #the single image behind streams/ and max_grids/
key_image_fallback=data_dir+'p63_vit_input_images/stephen/0111.png'   #un-cropped original, if frames.json is missing

## ---- Geometry shared with general_network_rendering (p25_35 / p13_30 lineage) ----
line_radius=0.18
cell_depth=0.1
pixel_dim=0.5
block_cell=0.48
base_depth=20.0                                  #world-unit depth of one block at depth_scale=1
G=37                                             #patch grid side (518 / 14)

still_hold=1.0


def vit_viz_block(a, z0, z_step, cell, pct=99, alpha=0.25, cell_z=cell_depth):
    """Positive part only, per-channel max-normalized, per-channel percentile-thresholded --
    same recipe as relu_viz_block in general_network_rendering, applied to a signed stream."""
    r=np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    vmax, thresh=layer_stats(r, pct)
    rn=r/vmax
    return conv_data_block(rn, z0, vmin=0.0, vmax=1.0, keep=(rn>thresh),
                           cell_size=cell, alpha=alpha, z_step=z_step, cell_z=cell_z)


## ---- Reading the p63 cache ----


## ---- Layout ----


## ---- The scene ----

class ViTStream(WelchScene):
    """40 residual-stream tensors of DINOv2 ViT-g/14 drawn as CNN-style activation blocks, then
    (optionally) collapsed into a 5x8 grid of the cached channel-max pngs."""

    model='plain'                     #'plain' or 'reg'
    first_idx=1                       #stream[1] = output of block 0. 0 would include the patch-embed input
    depth_pool=4                      #1536 -> 384 channels
    pool='max'                        #'max' or 'mean' over each group of depth_pool channels
    max_blocks=None                   #debug: only the first N blocks

    #Layout (same meaning as the CNN scenes)
    depth_scale=0.5                   #horizontal scale: block depth = base_depth*depth_scale world units
    layer_spacing=15.0 #3.0                 #world units between consecutive blocks
    cell=block_cell                   #face cell size; face width = 37*cell

    #Voxel selection
    act_pct=99                        #keep the top (100-pct)% of cells per channel. 40x384x37x37 cells:
    act_alpha=0.90                    # pct=99 -> ~210k voxels, pct=97 -> ~630k. Start high.

    #Input image block at the front of the chain
    show_image=True
    image_scale=1.5                   #image face width / block face width
    image_opacity=0.6
    image_depth=3*pixel_dim
    image_path=None                   #None -> cropped frame from stephen/input via frames.json (key_input_png)

    #Presentation
    fade_in=False
    fade_in_time=6.0
    default_view=None                 #(theta, phi, gamma, center, height); None -> auto side view
    do_collapse=False

    #Collapse-to-grid
    grid_rows=5
    grid_cols=8
    grid_cell=None                    #png width in the grid; None -> same as the block face (no rescale)
    grid_gap=None                     #None -> 0.15*grid_cell
    grid_image_scale=1.0              #input image width in the grid / grid_cell
    grid_image_gap=None               #space between the image and the first column; None -> 2*grid_gap
    grid_center_x=None                #None -> center of the network chain
    grid_y=0.0                        #the vertical plane the grid lives in
    grid_border=True
    grid_image_opacity=1.0
    png_upsample=8                    #nearest-neighbor upsample of the 37x37 pngs (None/1 -> off)
    collapse_factor=0.02              #how flat the block gets before the png takes over
    rotate_time=3.0
    collapse_time=3.0
    move_time=4.0
    lag_ratio=0.03                    #stagger between blocks within each stage
    arrows_svg=None                   #path to an Illustrator svg drawn over export_layout_svg's output
    arrows_color=None                 #None -> keep the svg's own colors
    arrows_stroke=3.0
    final_view=None                   #None -> auto front view framing the grid

    # ------------------------------------------------------------------

    def load(self):
        meta=load_meta()
        self.n_skip=meta['n_skip'][self.model]
        stream=load_stream(self.model)
        tensors=stream_tensors(stream, self.n_skip, self.first_idx, self.depth_pool, self.pool,
                               self.max_blocks)
        self.face_w=G*self.cell
        if self.show_image:
            self.image_w=self.image_scale*self.face_w
            hw=0.5*self.image_w
            self.image_bounds=(-hw, hw, -hw, hw, 0.0, self.image_depth)
            z_start=self.image_depth+self.layer_spacing
        else:
            z_start=0.0
        self.layers, self.total_z=block_layout(tensors, self.depth_scale, self.layer_spacing,
                                               self.cell, z_start)
        for L in self.layers:
            L['stream_idx']=self.first_idx+L['idx']
            L['png']=max_grid_png(self.model, L['stream_idx'])
        return self.layers

    def describe(self):
        n_vox=0
        print(f'\n{self.model}: {len(self.layers)} blocks, chain spans x=0 -> {self.total_z:.1f}, '
              f'face {self.face_w:.1f} wide')
        print(f'{"idx":>4}  {"name":10s} {"shape":16s} {"x0":>8} {"x1":>8}  voxels  png')
        for L, b in zip(self.layers, self.blocks):
            n_vox+=len(b.centers)
            print(f"{L['idx']:>4}  {L['name']:10s} {str(L['data'].shape):16s} "
                  f"{L['z0']:8.1f} {L['z1']:8.1f}  {len(b.centers):6d}  {os.path.basename(L['png'])}")
        print(f'total voxels: {n_vox}\n')

    def build_blocks(self):
        blocks, borders=[], []
        for L in self.layers:
            cz=min(cell_depth, 0.8*L['z_step'])
            blk, _=vit_viz_block(L['data'], L['z0'], L['z_step'], L['cell'],
                                 pct=self.act_pct, alpha=self.act_alpha, cell_z=cz)
            blocks.append(orient(blk))
            borders.append(orient(prism(*L['bounds'], CHILL_BROWN, line_radius)))
        return blocks, borders

    def build_image(self):
        b=self.image_bounds
        path=self.image_path or key_input_png()
        print(f'input image: {path}  (exists={os.path.exists(path)})', flush=True)
        img=gx.ImageMobject(path)
        print('input image loaded', flush=True)
        img.stretch_to_fit_width(b[1]-b[0])
        img.stretch_to_fit_height(b[3]-b[2])
        img.set_opacity(self.image_opacity)
        img.move_to([0, 0, 0.5*(b[4]+b[5])])
        return orient(img), orient(prism(*b, CHILL_BROWN, line_radius))

    def build(self):
        self.load()
        self.blocks, self.borders=self.build_blocks()
        self.describe()
        if self.show_image:
            self.img, self.image_border=self.build_image()
        else:
            self.img=self.image_border=None
        self.net=gx.Group(*self.blocks)
        return self

    def view(self):
        if self.default_view is not None:
            return self.default_view
        L=self.total_z
        return (2, 57, 0, (0.5*L, 0.0, 0.0), max(60.0, 0.95*L))

    # ---- grid geometry (world coords; grid plane is y=grid_y, x right, z up) ----

    def grid_geometry(self):
        cell=self.face_w if self.grid_cell is None else self.grid_cell
        gap=0.15*cell if self.grid_gap is None else self.grid_gap
        img_w=self.grid_image_scale*cell
        img_gap=2*gap if self.grid_image_gap is None else self.grid_image_gap
        grid_w=self.grid_cols*cell+(self.grid_cols-1)*gap
        grid_h=self.grid_rows*cell+(self.grid_rows-1)*gap
        comp_w=grid_w+(img_w+img_gap if self.show_image else 0.0)
        comp_h=max(grid_h, img_w if self.show_image else 0.0)
        cx=0.5*self.total_z if self.grid_center_x is None else self.grid_center_x
        x_left=cx-0.5*comp_w
        z_top=0.5*comp_h
        grid_left=x_left+(img_w+img_gap if self.show_image else 0.0)
        slots=[]
        for k in range(self.grid_rows*self.grid_cols):
            r, c=divmod(k, self.grid_cols)
            slots.append(np.array([grid_left+c*(cell+gap)+0.5*cell, self.grid_y,
                                   z_top-r*(cell+gap)-0.5*cell]))
        image_slot=np.array([x_left+0.5*img_w, self.grid_y, z_top-0.5*img_w])
        return dict(cell=cell, gap=gap, img_w=img_w, slots=slots, image_slot=image_slot,
                    bbox=(x_left, x_left+comp_w, -z_top, z_top), center=np.array([cx, self.grid_y, 0.0]),
                    width=comp_w, height=comp_h)

    def front_view(self):
        if self.final_view is not None:
            return self.final_view
        g=self.grid_geometry()
        h=max(1.2*g['height'], 1.1*g['width']*9/16)
        return (0, 90, 0, tuple(g['center']), h)

    def export_layout_svg(self, path=None, scale=10.0):
        """Write an svg of the final layout (frame rect + image slot + 40 numbered slots) to draw
        arrows over in Illustrator. Keep the outer 'frame' rect in the file you save back out (make
        it invisible if you like) so the import lands exactly; delete the slot rects or not."""
        g=self.grid_geometry()
        x0, x1, z0, z1=g['bbox']
        W, H=(x1-x0)*scale, (z1-z0)*scale
        rects=[]
        def rect(cx, cz, w, rid):
            x=(cx-0.5*w-x0)*scale
            y=(z1-(cz+0.5*w))*scale
            rects.append(f'<rect id="{rid}" x="{x:.2f}" y="{y:.2f}" width="{w*scale:.2f}" '
                         f'height="{w*scale:.2f}" fill="none" stroke="#948979" stroke-width="1"/>')
        if self.show_image:
            s=g['image_slot']; rect(s[0], s[2], g['img_w'], 'input')
        for k, s in enumerate(g['slots'][:len(self.layers)]):
            rect(s[0], s[2], g['cell'], f'layer_{self.layers[k]["stream_idx"]:02d}')
        svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}" height="{H:.2f}" '
             f'viewBox="0 0 {W:.2f} {H:.2f}">\n'
             f'<rect id="frame" x="0" y="0" width="{W:.2f}" height="{H:.2f}" fill="none" '
             f'stroke="#888888" stroke-width="0.5"/>\n'+'\n'.join(rects)+'\n</svg>\n')
        path=path or os.path.join(cache_dir, f'p63_grid_layout_{self.model}.svg')
        open(path, 'w').write(svg)
        print(f'layout svg -> {path}  ({W:.0f}x{H:.0f}, scale {scale} px per world unit)')
        return path

    def load_arrows_svg(self):
        """Illustrator svg (drawn over export_layout_svg's file, frame rect kept) into the grid plane."""
        if self.arrows_svg is None:
            return None
        g=self.grid_geometry()
        svg=gx.SVGMobject(self.arrows_svg)
        if self.arrows_color is not None:
            svg.set_stroke(self.arrows_color, width=self.arrows_stroke)
            svg.set_fill(self.arrows_color, opacity=1.0)
        else:
            svg.set_stroke(width=self.arrows_stroke)
        svg.scale_to_fit_width(g['width'])
        svg.move_to(gx.ORIGIN)
        to_front_plane(svg, y=self.grid_y-0.01)               #a hair in front of the pngs
        svg.move_to([g['center'][0], self.grid_y-0.01, 0.0])
        return svg

    # ---- collapse animation ----

    def build_grid_pngs(self, g):
        """One ImageMobject per block, sized to the grid cell, upright in the grid plane, parked
        on its slot (where the block's output face will be sitting when it collapses)."""
        pngs=[]
        for k, L in enumerate(self.layers):
            im=gx.ImageMobject(upsampled_png(L['png'], self.png_upsample))
            im.scale_to_fit_width(g['cell'])
            im.stretch_to_fit_height(g['cell'])
            to_front_plane(im, y=self.grid_y)
            im.move_to(g['slots'][k])
            pngs.append(im)
        return pngs

    def collapse_to_grid(self):
        """Stage A: the 3D blocks glide into the 5x8 arrangement while the camera tilts up to the
        front view. Stage B: each block spins so its output face sits on the grid plane, facing the
        camera. Stage C: each block squashes onto that face while the max-grid png fades in."""
        g=self.grid_geometry()
        pngs=self.build_grid_pngs(g)
        s_blk=g['cell']/self.face_w                              #uniform scale applied during the glide
        back=np.array([0.0, 1.0, 0.0])                          #+y = away from the camera

        # A block spinning -90 deg about its center sends its output face (x=c+d/2) to y=c_y-d/2, so
        # parking the center d/2 behind the grid plane puts the face exactly on the plane.
        items=[]                                                #(mob, border, start_center, target_center, scale, face_w)
        for k, (blk, bdr, L) in enumerate(zip(self.blocks, self.borders, self.layers)):
            c=np.array([0.5*(L['z0']+L['z1']), 0.0, 0.0])
            items.append((blk, bdr, c, g['slots'][k]+back*0.5*s_blk*L['depth'], s_blk, g['cell']))
        if self.show_image:
            s_img=g['img_w']/self.image_w
            ic=np.array([0.5*self.image_depth, 0.0, 0.0])
            items.append((self.img, self.image_border, ic,
                          g['image_slot']+back*0.5*s_img*self.image_depth, s_img, g['img_w']))

        # Stage A: glide into the grid layout + camera to the front view
        glides=[gx.AnimationGroup(GlideTo(m, t, start=c, scale=s), GlideTo(b, t, start=c, scale=s))
                for m, b, c, t, s, w in items]
        self.play(self.animate_view(*self.front_view()),
                  gx.LaggedStart(*glides, lag_ratio=self.lag_ratio), run_time=self.move_time)

        # Stage B: spin about the vertical axis so every output face points at the camera
        spins=[gx.AnimationGroup(SpinTo(m, -90*gx.DEGREES, gx.OUT, t), SpinTo(b, -90*gx.DEGREES, gx.OUT, t))
               for m, b, c, t, s, w in items]
        self.play(gx.LaggedStart(*spins, lag_ratio=self.lag_ratio), run_time=self.rotate_time)

        # Stage C: squash each block onto its front face (on the grid plane) as the png fades in
        squashes=[]
        for k, (m, b, c, t, s, w) in enumerate(items):
            face=np.array([t[0], self.grid_y, t[2]])
            anims=[SquashTo(m, self.collapse_factor, 1, face), SquashTo(b, self.collapse_factor, 1, face)]
            if k<len(pngs):
                anims.append(gx.FadeIn(pngs[k]))
            else:
                anims.append(self.img.animate.set_opacity(self.grid_image_opacity))
            squashes.append(gx.AnimationGroup(*anims))
        self.play(gx.LaggedStart(*squashes, lag_ratio=self.lag_ratio), run_time=self.collapse_time)

        # swap the squashed prisms for clean squares on the grid plane; drop the voxel blocks
        self.remove(*self.blocks, *self.borders)
        if self.show_image:
            self.remove(self.image_border)
            self.img.move_to(g['image_slot'])                  #squashed to ~0 depth; snap onto the plane
        outlines=[]
        if self.grid_border:
            for m, b, c, t, s, w in items:
                o=front_square(t[0], t[2], w, self.grid_y, CHILL_BROWN, line_radius)
                outlines.append(o)
                self.add(o)

        self.grid_pngs=pngs
        self.grid_outlines=outlines
        arrows=self.load_arrows_svg()
        if arrows is not None:
            self.play(gx.Write(arrows), run_time=2.0)
            self.arrows=arrows

    def construct(self):

        svgs_to_skip=[]
        svg_files=list(sorted(Path(svg_dir).glob('*.svg')))
        all_svgs=gx.Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=gx.SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        self.wait()

        np.random.seed(52)
        for i in [1, 2, 3]:
            for o in all_svgs[i]:
                o.set_opacity(np.random.rand(1,1).item())

        # all_svgs.set_color(CHILL_BROWN)

        self.add(all_svgs)

        self.build()
        self.set_view(*self.view())
        # if self.show_image:
        #     self.add(self.img, self.image_border)

        pairs=list(zip(self.blocks, self.borders))
        if self.fade_in:
            self.wait(1)
            fades=[gx.AnimationGroup(gx.FadeIn(b), gx.FadeIn(p)) for b, p in pairs]
            self.play(gx.LaggedStart(*fades, lag_ratio=1.0), run_time=self.fade_in_time)
        else:
            for b, p in pairs:
                self.add(b, p)

        all_svgs.scale(4.5)
        all_svgs.rotate(90*gx.DEGREES, [1, 0, 0])
        all_svgs.move_to([33.5, -4, 14])

        spacing=25.0
        all_transformer_layers=gx.Group()
        for l in range(39):
            tmp=all_svgs.copy()
            tmp.shift([l*spacing, 0, 0])
            all_transformer_layers.add(tmp)
        self.add(all_transformer_layers)

        # self.remove(all_transformer_layers)


        self.set_view(-3, 88, 0, (np.float32(38.03), np.float32(-47.99), np.float32(14.99)), 47.82)


        self.wait()


        # self.collapse_to_grid()
        self.set_view(-1, 90, 0, (np.float32(125.36), np.float32(6.09), np.float32(2.08)), 118.11)

        self.wait(still_hold)


        if self.do_collapse:
            self.collapse_to_grid()
            self.wait(still_hold)

class P63_Plain(ViTStream):
    """Step 1: tune depth_scale / layer_spacing / act_pct here."""
    model='plain'
    depth_scale=0.5
    layer_spacing=3.0


class P63_PlainCollapse(P63_Plain):
    """Step 2: blocks -> 5x8 grid of max_grids/plain/layer_01..40.png."""
    do_collapse=True
    # arrows_svg=cache_dir+'p63_arrows_plain.svg'   #after drawing over export_layout_svg()'s file


class P63_Reg(ViTStream):
    model='reg'
    depth_scale=0.5
    layer_spacing=3.0