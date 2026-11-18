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
from pathlib import Path
from PIL import Image


data_dir = str(ACTIVATION_ROOT) + '/'
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


## ---- Cheap in-place animations for big meshes ----
# Rotate/Transform copy the whole mobject per frame; a 40-block chain of VoxelBlocks is millions of
# vertices, so these apply incremental rotations / stretches to the live points instead.


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
    layer_spacing=3.0                 #world units between consecutive blocks
    cell=block_cell                   #face cell size; face width = 37*cell

    #Voxel selection
    act_pct=99                        #keep the top (100-pct)% of cells per channel. 40x384x37x37 cells:
    act_alpha=0.25                    # pct=99 -> ~210k voxels, pct=97 -> ~630k. Start high.

    #Input image block at the front of the chain
    show_image=True
    image_scale=1.5                   #image face width / block face width
    image_opacity=0.6
    image_depth=3*pixel_dim
    image_path=None                   #None -> cropped frame from stephen/input via frames.json (key_input_png)

    #Transformer-block graphic (Illustrator svgs) standing in the gap between consecutive blocks
    svg_dir=None                      #folder of the exported svgs, sorted by name; None -> no graphic
    svg_skip=[]                       #indices (in sorted order) to leave out
    svg_random_layers=[1, 2, 3]       #group indices whose paths get random opacities
    svg_seed=52
    svg_reseed=False                  #True -> a different random pattern in every copy
    svg_width=None                    #None -> 0.8*layer_spacing
    svg_z=0.0                         #vertical center of the graphic relative to the block centers
    svg_y=0.0                         #depth plane; 0 = through the block centers
    svg_x_offset=0.0
    svg_before_first=False            #also put one between the input image and block 0
    svg_depth_test=False

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
    image_spin_margin=2.0             #clearance between the flattened image and block 0's swing
    collapse_order='collapse_first'   #'collapse_first': spin+flatten in place, then fly to the grid
                                      #'grid_first': fly to the grid as 3D blocks, then spin+flatten
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

    def load_block_graphic(self):
        files=sorted(Path(self.svg_dir).glob('*.svg'))
        group=gx.Group()
        for i, f in enumerate(files):
            if i in self.svg_skip:
                continue
            svg=gx.SVGMobject(str(f))
            group.add(svg[1:])                                  #drop Illustrator's background rect
        return group

    def randomize_svg_opacity(self, graphic, seed):
        rng=np.random.default_rng(seed)
        for i in self.svg_random_layers:
            if i<len(graphic):
                for o in graphic[i]:
                    o.set_opacity(rng.random())

    def gap_centers(self):
        """x of the middle of every gap the graphic goes in."""
        xs=[]
        if self.show_image and self.svg_before_first:
            xs.append(0.5*(self.image_depth+self.layers[0]['z0']))
        for a, b in zip(self.layers[:-1], self.layers[1:]):
            xs.append(0.5*(a['z1']+b['z0']))
        return xs

    def build_block_graphics(self):
        """One copy of the graphic per gap, upright in the vertical plane y=svg_y (seen face-on
        from the side view), centered in x on the gap."""
        base=self.load_block_graphic()
        w=0.8*self.layer_spacing if self.svg_width is None else self.svg_width
        base.scale_to_fit_width(w)
        graphics=gx.Group()
        for k, x in enumerate(self.gap_centers()):
            gfx=base.copy()
            self.randomize_svg_opacity(gfx, self.svg_seed+(k if self.svg_reseed else 0))
            gfx.move_to(gx.ORIGIN)
            to_front_plane(gfx, y=self.svg_y)
            gfx.move_to([x+self.svg_x_offset, self.svg_y, self.svg_z])
            graphics.add(gfx)
        return graphics

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

    def build_grid_pngs(self, width, positions):
        """One ImageMobject per block, upright in the grid plane, at the given positions."""
        pngs=[]
        for L, pos in zip(self.layers, positions):
            im=gx.ImageMobject(upsampled_png(L['png'], self.png_upsample))
            im.scale_to_fit_width(width)
            im.stretch_to_fit_height(width)
            to_front_plane(im, y=self.grid_y)
            im.move_to(pos)
            pngs.append(im)
        return pngs

    def fade_graphics(self):
        if getattr(self, 'block_graphics', None) is not None:
            return [gx.FadeOut(self.block_graphics)]
        return []

    def finish_grid(self, pngs, outlines):
        self.grid_pngs=pngs
        self.grid_outlines=outlines
        arrows=self.load_arrows_svg()
        if arrows is not None:
            self.play(gx.Write(arrows), run_time=2.0)
            self.arrows=arrows

    def collapse_to_grid(self):
        if self.collapse_order=='grid_first':
            self.collapse_grid_first()
        else:
            self.collapse_in_place()

    def image_spin_shift(self):
        """The input image is wider than it is deep, so once it spins flat it reaches further along
        x than its prism did -- right into block 0's own swing. Shift it left during the spin by
        just enough to clear: [dx, 0, 0], dx <= 0."""
        L0=self.layers[0]
        block_reach=L0['z0']+0.5*L0['depth']-0.5*self.face_w      #leftmost point of flat block 0
        image_reach=0.5*self.image_depth+0.5*self.image_w         #rightmost point of the flat image
        dx=min(0.0, block_reach-image_reach-self.image_spin_margin)
        return np.array([dx, 0.0, 0.0])

    def collapse_in_place(self):
        """Stage A: every block spins -90 deg about its own center so its output face points at the
        camera. Stage B: each block squashes flat in place while its max-grid png fades in. Stage C:
        the flat images fly to their 5x8 slots as the camera tilts up to the front view."""
        g=self.grid_geometry()
        centers=[np.array([0.5*(L['z0']+L['z1']), 0.0, 0.0]) for L in self.layers]
        pngs=self.build_grid_pngs(self.face_w, centers)

        # Stage A: spin in place (transformer-block graphics fade out here). The image also drifts
        # left so its flattened width clears block 0's swing.
        spins=[]
        if self.show_image:
            ic=np.array([0.5*self.image_depth, 0.0, 0.0])
            dv=self.image_spin_shift()
            ic_final=ic+dv
            spins.append(gx.AnimationGroup(SpinTo(self.img, -90*gx.DEGREES, gx.OUT, ic, shift=dv),
                                        SpinTo(self.image_border, -90*gx.DEGREES, gx.OUT, ic, shift=dv)))
        for blk, bdr, c in zip(self.blocks, self.borders, centers):
            spins.append(gx.AnimationGroup(SpinTo(blk, -90*gx.DEGREES, gx.OUT, c),
                                        SpinTo(bdr, -90*gx.DEGREES, gx.OUT, c)))
        self.play(gx.LaggedStart(*spins, lag_ratio=self.lag_ratio), *self.fade_graphics(),
                  run_time=self.rotate_time)

        # Stage B: squash flat about each center while the png fades in there
        squashes=[]
        for blk, bdr, png, c in zip(self.blocks, self.borders, pngs, centers):
            squashes.append(gx.AnimationGroup(SquashTo(blk, self.collapse_factor, 1, c),
                                           SquashTo(bdr, self.collapse_factor, 1, c),
                                           gx.FadeIn(png)))
        if self.show_image:
            squashes.insert(0, gx.AnimationGroup(SquashTo(self.img, self.collapse_factor, 1, ic_final),
                                              SquashTo(self.image_border, self.collapse_factor, 1, ic_final)))
        self.play(gx.LaggedStart(*squashes, lag_ratio=self.lag_ratio), run_time=self.collapse_time)

        # swap the squashed prisms for clean squares; drop the voxel blocks
        self.remove(*self.blocks, *self.borders)
        outlines=[]
        if self.grid_border:
            for c in centers:
                o=front_square(c[0], c[2], self.face_w, self.grid_y, CHILL_BROWN, line_radius)
                outlines.append(o)
                self.add(o)
        img_outline=None
        if self.show_image:
            self.remove(self.image_border)
            ic2=np.array([ic_final[0], self.grid_y, ic_final[2]])
            self.img.move_to(ic2)
            if self.grid_border:
                img_outline=front_square(ic2[0], ic2[2], self.image_w, self.grid_y, CHILL_BROWN, line_radius)
                outlines.append(img_outline)
                self.add(img_outline)

        # Stage C: fly to the 5x8 grid while the camera tilts up to the front view
        moves=[]
        if self.show_image:
            t=g['image_slot']
            anims=[self.img.animate.move_to(t).scale_to_fit_width(g['img_w']).set_opacity(self.grid_image_opacity)]
            if img_outline is not None:
                anims.append(img_outline.animate.move_to(t).scale_to_fit_width(g['img_w']))
            moves.append(gx.AnimationGroup(*anims))
        for k, png in enumerate(pngs):
            t=g['slots'][k]
            anims=[png.animate.move_to(t).scale_to_fit_width(g['cell'])]
            if self.grid_border:
                anims.append(outlines[k].animate.move_to(t).scale_to_fit_width(g['cell']))
            moves.append(gx.AnimationGroup(*anims))
        self.play(self.animate_view(*self.front_view()),
                  gx.LaggedStart(*moves, lag_ratio=self.lag_ratio), run_time=self.move_time)

        self.finish_grid(pngs, outlines)

    def collapse_grid_first(self):
        """Stage A: the 3D blocks glide into the 5x8 arrangement while the camera tilts up to the
        front view. Stage B: each block spins so its output face sits on the grid plane, facing the
        camera. Stage C: each block squashes onto that face while the max-grid png fades in."""
        g=self.grid_geometry()
        pngs=self.build_grid_pngs(g['cell'], g['slots'])
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
                  gx.LaggedStart(*glides, lag_ratio=self.lag_ratio), *self.fade_graphics(),
                  run_time=self.move_time)

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

        self.finish_grid(pngs, outlines)

    def construct(self):
        self.build()
        self.set_view(*self.view())
        if self.show_image:
            self.add(self.img, self.image_border)

        self.block_graphics=self.build_block_graphics() if self.svg_dir else None
        gfx=list(self.block_graphics) if self.block_graphics is not None else []
        if self.svg_before_first and self.show_image and gfx:
            gfx=[None]+gfx                                        #so gfx[k] is the graphic AFTER block k

        pairs=list(zip(self.blocks, self.borders))
        if self.fade_in:
            self.wait(1)
            fades=[]
            for k, (b, p) in enumerate(pairs):
                anims=[gx.FadeIn(b), gx.FadeIn(p)]
                if k<len(gfx) and gfx[k] is not None:
                    anims.append(gx.FadeIn(gfx[k]))
                fades.append(gx.AnimationGroup(*anims))
            if gfx and gfx[0] is None:
                self.add(self.block_graphics[0])
            self.play(gx.LaggedStart(*fades, lag_ratio=1.0), run_time=self.fade_in_time)
        else:
            for b, p in pairs:
                self.add(b, p)
            if self.block_graphics is not None:
                self.add(self.block_graphics)

        self.wait(still_hold)
        if self.do_collapse:
            self.collapse_to_grid()
            self.wait(still_hold)

class P63_Plain(ViTStream):
    """Step 1: tune depth_scale / layer_spacing / act_pct here."""
    model='plain'
    depth_scale=0.5
    layer_spacing=15.0
    # svg_dir=data_dir+'p63_transformer_block_svgs/'    #the 4 Illustrator exports


class P63_PlainCollapse(P63_Plain):
    """Step 2: blocks -> 5x8 grid of max_grids/plain/layer_01..40.png."""
    do_collapse=True
    # arrows_svg=cache_dir+'p63_arrows_plain.svg'   #after drawing over export_layout_svg()'s file


class P63_Reg(ViTStream):
    model='reg'
    depth_scale=0.5
    layer_spacing=3.0