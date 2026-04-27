from manimlib import *
from tqdm import tqdm
from pathlib import Path
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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/p53_60/')

## SWITCH BACK TO FULL SIZE IMAGES FOR FINAL RENDER
# img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/img_pairs'

#TINY IMAGES FOR DEV
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/img_pairs_tiny'


def make_correlation_matrix(
    N=20,
    noise_scale=0.18,
    total_size=3.0,
    center=ORIGIN,
    seed=42,
    line_stroke_width=1.0,
):
    """
    Returns:
        M                    : (N, N) symmetric numpy matrix (identity + noise)
        grid_lines           : VGroup of thin magenta cell outlines
        diagonal_squares     : VGroup of N magenta squares on the diagonal
        off_diagonal_squares : VGroup of N*(N-1) magenta squares off the diagonal
                               (opacities = |M[i, j]|)
    """
    rng = np.random.default_rng(seed)
    M = np.eye(N) + noise_scale * rng.standard_normal((N, N))
    M = (M + M.T) / 2.0           # symmetrize across the diagonal
    # np.fill_diagonal(M, 1.0)      # clean ones on the diagonal
    M = np.clip(M, -1.0, 1.0)

    cell = total_size / N
    half = total_size / 2.0

    # --- grid lines (N+1 verticals + N+1 horizontals) ---
    grid_lines = VGroup()
    for k in range(N + 1):
        off = -half + k * cell
        v = Line([off, -half, 0], [off,  half, 0], stroke_width=line_stroke_width)
        h = Line([-half,   off, 0], [ half,  off, 0], stroke_width=line_stroke_width)
        v.set_color(MAGENTA)
        h.set_color(MAGENTA)
        grid_lines.add(v, h)

    # --- filled squares, split into diagonal vs off-diagonal ---
    diagonal_squares = VGroup()
    off_diagonal_squares = VGroup()
    for i in range(N):                # row (top -> bottom)
        for j in range(N):            # col (left -> right)
            x = -half + (j + 0.5) * cell
            y =  half - (i + 0.5) * cell
            sq = Square(side_length=cell)
            sq.move_to([x, y, 0])
            sq.set_stroke(width=0)
            sq.set_fill(MAGENTA, opacity=float(abs(M[i, j])))
            if i == j:
                diagonal_squares.add(sq)
            else:
                off_diagonal_squares.add(sq)

    full = VGroup(off_diagonal_squares, diagonal_squares, grid_lines)
    full.move_to(center)              # squares behind, lines on top

    return M, grid_lines, diagonal_squares, off_diagonal_squares




class P53_60(InteractiveScene):
    def construct(self):


        imgs=Group()
        for i in range(1,23):
            imgs.add(ImageMobject(str(img_dir+'/img_pairs-'+str(i).zfill(2)+'.png')))


        svgs_to_skip=[0, 1, 2, 3, 4]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        

        embedding_network_1=all_svgs[1]
        embedding_network_2=embedding_network_1.copy()

        embedding_network_1.move_to([-3, -0.2, 0 ])
        embedding_network_2.move_to([3, -0.2, 0 ])

        imgs[0].scale(0.5)
        imgs[0].next_to(embedding_network_1, DOWN, buff=0.2)
        border_0 = SurroundingRectangle(imgs[0], color=CHILL_BROWN, buff=0)
        border_0.set_stroke(width=2, opacity=1.0)

        img_1_copy=imgs[1].copy()

        img_1_copy.scale(0.5)
        img_1_copy.next_to(embedding_network_2, DOWN, buff=0.2)
        border_1 = SurroundingRectangle(img_1_copy, color=CHILL_BROWN, buff=0)
        border_1.set_stroke(width=2, opacity=1.0)

        self.wait()
        self.play(Write(embedding_network_1), 
                  Write(embedding_network_2),
                  FadeIn(imgs[0]),
                  FadeIn(border_0),
                  FadeIn(img_1_copy),
                  FadeIn(border_1),
                  run_time=7)

        # self.add(embedding_network_1, embedding_network_2)
        # self.add(imgs[0], img_1_copy)
        # self.add(border_0, border_1)


        # self.add(all_svgs[2])
        all_svgs[2].move_to([-3.01, 1.7, 0])
        all_svgs[3].next_to(all_svgs[2], UP, buff=0.2)
        all_svgs[4].move_to([-4.5, 0.65, 0]) 

        self.wait()
        self.play(FadeIn(all_svgs[2]),
                  embedding_network_1[75].animate.set_color(YELLOW),
                  embedding_network_1[83].animate.set_color(YELLOW),
                  embedding_network_1[87].animate.set_color(YELLOW),
                )

        self.wait()
        self.play(Write(all_svgs[3]), run_time=3)

        self.wait()
        self.play(Write(all_svgs[4]), run_time=3)

        

        image_border_group_0=Group(imgs[0], border_0)
        image_border_group_1=Group(img_1_copy, border_1)

        embedding_network_1a=embedding_network_1.copy()
        embedding_network_1a[75].set_color(CHILL_BROWN)
        embedding_network_1a[83].set_color(CHILL_BROWN)
        # embedding_network_1a[87].set_color(CHILL_BROWN)

        embedding_network_1a.rotate(-90*DEGREES, [0, 0, 1])
        embedding_network_1a.move_to([-3, 1.5, 0])
        embedding_network_1a.scale(0.85)

        embedding_network_2b=embedding_network_2.copy()
        embedding_network_2b.rotate(-90*DEGREES, [0, 0, 1])
        embedding_network_2b.move_to([-3, -2.0, 0])
        embedding_network_2b.scale(0.85)

        self.wait()
        self.remove(all_svgs[4], all_svgs[3], all_svgs[2])
        self.play(
                  #FadeOut(all_svgs[4]),
                  #FadeOut(all_svgs[3]),
                  #FadeOut(all_svgs[2]),
                  Transform(embedding_network_1, embedding_network_1a),
                  Transform(embedding_network_2, embedding_network_2b),
                  image_border_group_0.animate.move_to([-5, 1.5, 0]),
                  image_border_group_1.animate.move_to([-5, -2.0, 0]),
                  run_time=5)

        # self.remove(all_svgs[4], all_svgs[3], all_svgs[2])
        # embedding_network_1[75].set_color(CHILL_BROWN)
        # embedding_network_1[83].set_color(CHILL_BROWN)
        # embedding_network_1[87].set_color(CHILL_BROWN)

        # embedding_network_1.rotate(-90*DEGREES, [0, 0, 1])
        # embedding_network_1.move_to([-3, 1.5, 0])
        # embedding_network_1.scale(0.85)

        # embedding_network_2.rotate(-90*DEGREES, [0, 0, 1])
        # embedding_network_2.move_to([-3, -1.5, 0])
        # embedding_network_2.scale(0.85)

        # image_border_group_0.move_to([-5, 1.5, 0])
        # image_border_group_1.move_to([-5, -1.5, 0])

        # image_border_group_0.next_to(embedding_network_1, LEFT, buff=0.2)
        # image_border_group_1.next_to(embedding_network_2, LEFT, buff=0.2)

        all_svgs[8].move_to([-1.65, 0.2, 0])

        # self.wait()
        # self.add(all_svgs[8][0])

        axes = Axes(
            x_range=(0, 21, 5),
            y_range=(-1.2, 1.2, 0.5),
            width=7.0,
            height=2.0,
            axis_config={
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_ticks": False,
                "include_tip": True,
                "tip_config": {
                    "width": 0.15,
                    "length": 0.15,
                },
            }
        )
        axes.move_to([2.4, 1.7, 0])
        # self.add(axes)

        # Explicit but semi-random y values
        y_values = [0.9, -0.6, 0.2, 0.5, -0.3, 0.7, 0.6, 0.1, 0.4, -0.5,
                    0.75, -0.25, 0.55, 0.7, 0.3, -0.45, -0.6, -0.15, 0.8, 0.35]
        x_values = list(range(1, 21))

        # Build dots and connecting lines up front
        dots = VGroup()
        lines = VGroup()
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            dot = Dot(axes.c2p(x, y), radius=0.06)
            dot.set_color(YELLOW)
            dots.add(dot)
            if i > 0:
                line = Line(
                    axes.c2p(x_values[i-1], y_values[i-1]),
                    axes.c2p(x, y),
                    stroke_width=2,
                )
                line.set_color(YELLOW)
                lines.add(line)

        self.wait()
        self.play(Write(all_svgs[8][0]), ShowCreation(axes), run_time=3)


        dashed_line_0 = DashedLine(
            # image_border_group_0.get_bottom(),
            [-0.73630941,  2.77874996,  0.        ],
            dots[0].get_top(),
            color=CHILL_BROWN,
            stroke_width=3,
            dash_length=0.03,
        )


        self.wait()
        self.play(
            image_border_group_0.animate.scale(0.4).next_to(dots[0], UP, buff=0.5),
            ShowCreation(dots[0]),
            ShowCreation(dashed_line_0),
            run_time=4
            )

        #Maybe put cat pic in both places at once? That might be a fine workflow here
        imgs[2].scale(0.5)
        border_2=SurroundingRectangle(imgs[2], color=CHILL_BROWN, buff=0)
        border_2.set_stroke(width=2, opacity=1.0)
        image_border_group_2=Group(imgs[2], border_2)
        image_border_group_2.move_to([-5, 1.5, 0])

        image_border_group_2_copy=image_border_group_2.copy()
        image_border_group_2_copy.scale(0.4).next_to(dots[1], DOWN, buff=0.2)

        # self.wait()
        # image_border_group_2_copy.get


        dashed_line_1 = DashedLine(image_border_group_2_copy.get_top(), dots[1].get_bottom(),
                                    color=CHILL_BROWN, stroke_width=3, dash_length=0.03)

        self.wait()
        self.play(ShowCreation(dots[1]), 
                  FadeIn(image_border_group_2), 
                  FadeIn(image_border_group_2_copy),
                  ShowCreation(lines[0]), 
                  ShowCreation(dashed_line_1), 
                  run_time=3)


        self.wait()
        self.remove(image_border_group_2)

        top_image_group=Group()
        dot_indices=[3, 4, 6, 7, 10, 11, 13, 16, 18]
        image_indices=[4, 6, 8, 10, 12, 14, 16, 18, 20]
        for i in range(len(dot_indices)):
            imgs[image_indices[i]].scale(0.4)
            border=SurroundingRectangle(imgs[image_indices[i]], color=CHILL_BROWN, buff=0)
            border.set_stroke(width=2, opacity=1.0)
            ibg=Group(imgs[image_indices[i]], border)
            if i % 2==0:
                ibg.scale(0.5).next_to(dots[dot_indices[i]], UP, buff=0.2)
                dashed_line = DashedLine(ibg.get_bottom(), dots[dot_indices[i]].get_top(),
                        color=CHILL_BROWN, stroke_width=3, dash_length=0.03)
            else:
                ibg.scale(0.5).next_to(dots[dot_indices[i]], DOWN, buff=0.2)
                dashed_line = DashedLine(ibg.get_top(), dots[dot_indices[i]].get_bottom(),
                        color=CHILL_BROWN, stroke_width=3, dash_length=0.03)
            ibg.add(dashed_line)
            top_image_group.add(ibg)
            self.add(ibg, dots[:(dot_indices[i]+1)], lines[:dot_indices[i]])
            self.wait(0.2)

        self.wait()


        y_values_2 = [0.8, -0.5, 0.25, 0.45, -0.1, 0.4, 0.4, 0.2, 0.5, -0.6,
                    0.7, -0.2, 0.45, 0.71, 0.21, -0.55, -0.3, -0.21, 0.7, 0.28]

        axes_2 = Axes(
            x_range=(0, 21, 5),
            y_range=(-1.2, 1.2, 0.5),
            width=7.0,
            height=2.0,
            axis_config={
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_ticks": False,
                "include_tip": True,
                "tip_config": {
                    "width": 0.15,
                    "length": 0.15,
                },
            }
        )
        axes_2.move_to([2.4, -1.8, 0])
        all_svgs[8][1].move_to([-1.624, -1.68, 0])
        all_svgs[8][1].set_color(RED)

        # Build dots and connecting lines up front
        dots_2 = VGroup()
        lines_2 = VGroup()
        for i, (x, y) in enumerate(zip(x_values, y_values_2)):
            dot = Dot(axes_2.c2p(x, y), radius=0.06)
            dot.set_color(RED)
            dots_2.add(dot)
            if i > 0:
                line = Line(
                    axes_2.c2p(x_values[i-1], y_values_2[i-1]),
                    axes_2.c2p(x, y),
                    stroke_width=2,
                )
                line.set_color(RED)
                lines_2.add(line)

        self.wait()
        self.play(ShowCreation(axes_2), 
                  embedding_network_2[87].animate.set_color(RED),
                  FadeIn(all_svgs[8][1]),
                  run_time=2
                 )

        self.wait()

        # Move dog picture over when we bring everything in!

        dashed_line_0b = DashedLine(
            # image_border_group_0.get_bottom(),
            [-0.73630941, -0.79833341,  0.        ],
            dots_2[0].get_top(),
            color=CHILL_BROWN,
            stroke_width=3,
            dash_length=0.03,
        )


        self.wait()
        self.play(
            image_border_group_1.animate.scale(0.4).next_to(dots_2[0], UP, buff=0.4),
            ShowCreation(dots_2[0]),
            ShowCreation(dashed_line_0b),
            run_time=2
            )

        self.wait()
        bottom_image_group=Group()
        dot_indices=[1, 3, 4, 6, 7, 10, 11, 13, 16, 18]
        image_indices=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        for i in range(len(dot_indices)):
            imgs[image_indices[i]].scale(0.4)
            border=SurroundingRectangle(imgs[image_indices[i]], color=CHILL_BROWN, buff=0)
            border.set_stroke(width=2, opacity=1.0)
            ibg=Group(imgs[image_indices[i]], border)
            if i % 2==1:
                ibg.scale(0.5).next_to(dots_2[dot_indices[i]], UP, buff=0.2)
                dashed_line = DashedLine(ibg.get_bottom(), dots_2[dot_indices[i]].get_top(),
                        color=CHILL_BROWN, stroke_width=3, dash_length=0.03)
            else:
                ibg.scale(0.5).next_to(dots_2[dot_indices[i]], DOWN, buff=0.2)
                dashed_line = DashedLine(ibg.get_top(), dots_2[dot_indices[i]].get_bottom(),
                        color=CHILL_BROWN, stroke_width=3, dash_length=0.03)
            ibg.add(dashed_line)
            bottom_image_group.add(ibg)

            self.add(ibg, dots_2[:(dot_indices[i]+1)], lines_2[:dot_indices[i]])
            self.wait(0.1)


        axis_0_group=Group(axes, dots, lines)
        axis_1_group=Group(axes_2, dots_2, lines_2)

        axis_0_group_copy=axis_0_group.copy()
        axis_1_group_copy=axis_1_group.copy()

        all_image_callouts=Group(image_border_group_1, dashed_line_0b, top_image_group, 
                    bottom_image_group, image_border_group_2_copy, dashed_line_1, 
                    image_border_group_0, dashed_line_0)

        self.wait()
        self.remove(all_image_callouts)
        self.play(
                axis_0_group.animate.move_to([2.4, -0.1, 0]),
                axis_1_group.animate.move_to([2.4, -0.1, 0]),
                self.frame.animate.reorient(0, 0, 0, (1.26, -0.25, 0.0), 6.70),
                run_time=5
                )

        #More into one curve temporarily
        y_values_average = (np.array(y_values)+np.array(y_values_2))/2.0
        dots_average_1 = VGroup()
        lines_average_1 = VGroup()
        for i, (x, y) in enumerate(zip(x_values, y_values_average)):
            dot = Dot(axes_2.c2p(x, y), radius=0.06)
            dots_average_1.add(dot)
            if i > 0:
                line = Line(
                    axes_2.c2p(x_values[i-1], y_values_average[i-1]),
                    axes_2.c2p(x, y),
                    stroke_width=2,
                )
                lines_average_1.add(line)

        dots_average_1.set_color(YELLOW).set_opacity(0.7)
        lines_average_1.set_color(YELLOW).set_opacity(0.7)
        dots_average_2=dots_average_1.copy()
        lines_average_2=lines_average_1.copy()
        dots_average_2.set_color(RED).set_opacity(0.7)
        lines_average_2.set_color(RED).set_opacity(0.7)


        self.wait()
        self.play(ReplacementTransform(dots,dots_average_1), 
                  ReplacementTransform(dots_2,dots_average_2), 
                  ReplacementTransform(lines,lines_average_1), 
                  ReplacementTransform(lines_2,lines_average_2), 
                  run_time=2)
        # self.remove(dots_2, dots, lines, lines_2)

        y_values_collapsed = np.ones(len(y_values))
        dots_collapsed_1 = VGroup()
        lines_collapsed_1 = VGroup()
        for i, (x, y) in enumerate(zip(x_values, y_values_collapsed)):
            dot = Dot(axes_2.c2p(x, y), radius=0.06)
            dots_collapsed_1.add(dot)
            if i > 0:
                line = Line(
                    axes_2.c2p(x_values[i-1], y_values_collapsed[i-1]),
                    axes_2.c2p(x, y),
                    stroke_width=2,
                )
                lines_collapsed_1.add(line)   
        
        dots_collapsed_1.set_color(YELLOW).set_opacity(0.7)
        lines_collapsed_1.set_color(YELLOW).set_opacity(0.7)
        dots_collapsed_2=dots_collapsed_1.copy()
        lines_collapsed_2=lines_collapsed_1.copy()
        dots_collapsed_2.set_color(RED).set_opacity(0.7)
        lines_collapsed_2.set_color(RED).set_opacity(0.7)

        self.wait()
        self.play(ReplacementTransform(dots_average_1, dots_collapsed_1), 
                  ReplacementTransform(dots_average_2, dots_collapsed_2), 
                  ReplacementTransform(lines_average_1, lines_collapsed_1), 
                  ReplacementTransform(lines_average_2, lines_collapsed_2), 
                  run_time=3)

        # Ok now I basically want to reverse the last few steps, 
        # all in one go

        self.wait()
        self.play(ReplacementTransform(axes, axis_0_group_copy[0]),
                  ReplacementTransform(dots_collapsed_1, axis_0_group_copy[1]),
                  ReplacementTransform(lines_collapsed_1, axis_0_group_copy[2]),
                  ReplacementTransform(axes_2, axis_1_group_copy[0]),
                  ReplacementTransform(dots_collapsed_2, axis_1_group_copy[1]),
                  ReplacementTransform(lines_collapsed_2, axis_1_group_copy[2]),
                  FadeOut(embedding_network_1[0]), #Remove little arrow going into net
                  FadeOut(embedding_network_2[0]),
                  # self.frame.animate.reorient(0, 0, 0, (1.19, -0.08, 0.0)), 
                  self.frame.animate.reorient(0, 0, 0, (1.48, 0.11, 0.0), 7.81),
                  run_time=5
                 )
        self.add(all_image_callouts)


        np.random.seed(5)
        y_values_net_1_neuron_2=-0.6*np.array(y_values)+np.random.randn(len(y_values))/3.5
        y_values_net_2_neuron_2=-0.6*np.array(y_values_2)+np.random.randn(len(y_values))/3.5

        dots_net_1_neuron_2 = VGroup()
        lines_net_1_neuron_2 = VGroup()
        for i, (x, y) in enumerate(zip(x_values, y_values_net_1_neuron_2)):
            dot = Dot(axes.c2p(x, y), radius=0.06)
            dot.set_color(GREEN)
            dots_net_1_neuron_2.add(dot)
            if i > 0:
                line = Line(
                    axes.c2p(x_values[i-1], y_values_net_1_neuron_2[i-1]),
                    axes.c2p(x, y),
                    stroke_width=2,
                )
                line.set_color(GREEN)
                lines_net_1_neuron_2.add(line)

        dots_net_2_neuron_2 = VGroup()
        lines_net_2_neuron_2 = VGroup()
        for i, (x, y) in enumerate(zip(x_values, y_values_net_2_neuron_2)):
            dot = Dot(axes_2.c2p(x, y), radius=0.06)
            dot.set_color(BLUE)
            dots_net_2_neuron_2.add(dot)
            if i > 0:
                line = Line(
                    axes_2.c2p(x_values[i-1], y_values_net_2_neuron_2[i-1]),
                    axes_2.c2p(x, y),
                    stroke_width=2,
                )
                line.set_color(BLUE)
                lines_net_2_neuron_2.add(line)


        # self.remove(top_image_group, image_border_group_2_copy, dashed_line_0, image_border_group_0, dashed_line_1)
        # self.remove(bottom_image_group, dashed_line_0b, image_border_group_1, dashed_line_1)
        
        self.wait()
        self.play(FadeOut(all_image_callouts), run_time=2)


        all_svgs[9][6:].move_to([-1.67, 1.54, 0])
        embedding_network_1[83].set_color(GREEN)
        self.wait(0.1)
        self.add(all_svgs[9][6:])
        self.wait(0.1)
        for i in range(len(dots_net_1_neuron_2)):
            self.add(dots_net_1_neuron_2[i])
            if i>0:
                self.add(lines_net_1_neuron_2[i-1])
            self.wait(0.1)

        # self.add(dots_net_1_neuron_2, lines_net_1_neuron_2)

        all_svgs[9][4:6].move_to([-1.67, -1.95, 0])
        all_svgs[9][4:6].set_color(BLUE)

        embedding_network_2[83].set_color(BLUE)
        self.wait(0.1)
        self.add(all_svgs[9][4:6])
        self.wait(0.1)
        for i in range(len(dots_net_2_neuron_2)):
            self.add(dots_net_2_neuron_2[i])
            if i>0:
                self.add(lines_net_2_neuron_2[i-1])
            self.wait(0.1)

        
        # self.add(dots_net_2_neuron_2, lines_net_2_neuron_2)

        # axis_0_group_copy[1].set_opacity(0.5)
        # axis_0_group_copy[2].set_opacity(0.5)
        # imgs[6].set_opacity(0.1)
        # self.remove(dashed_line_0b)



        self.wait()
        self.play(FadeOut(all_svgs[9][6:]), 
                  FadeOut(dots_net_1_neuron_2),
                  FadeOut(lines_net_1_neuron_2),
                  FadeOut(all_svgs[8][1]),
                  FadeOut(axis_1_group_copy[1]), #Dots 
                  FadeOut(axis_1_group_copy[2]), #lines
                  run_time=2
                  )

        self.wait()

        # self.remove(all_svgs[9][4:6], all_svgs[8][0])


        #axis_0_group_copy
        axis_1_group_b=Group(axis_1_group_copy[0], dots_net_2_neuron_2, lines_net_2_neuron_2)

        self.wait()
        self.play(
                axis_0_group_copy.animate.move_to([2.4, -0.1, 0]),
                axis_1_group_b.animate.move_to([2.4, -0.1, 0]),
                FadeOut(all_svgs[9][4:6]),
                FadeOut(all_svgs[8][0]),
                # self.frame.animate.reorient(0, 0, 0, (1.26, -0.25, 0.0), 6.70),
                run_time=3
                )

        #Ok now scaled versions! Factor of two here to make things land like i want on the axes
        
        helper_axes = Axes(
            x_range=(0, 21, 5),
            y_range=(-1.2, 1.2, 0.5),
            width=7.0,
            height=2.0,
            axis_config={
                "color": CHILL_BROWN,
                "stroke_width": 2,
                "include_ticks": False,
                "include_tip": True,
                "tip_config": {"width": 0.15, "length": 0.15},
            },
        )
        helper_axes.move_to([2.4, -0.1, 0])

        norm_1=np.linalg.norm(y_values)/3
        norm_2=np.linalg.norm(y_values_net_2_neuron_2)/3

        y_normed_1=y_values/norm_1
        y_normed_2=y_values_net_2_neuron_2/norm_2

        dots_norm_1 = VGroup()
        lines_norm_1 = VGroup()
        for i, (x, y) in enumerate(zip(x_values, y_normed_1)):
            dot = Dot(helper_axes.c2p(x, y), radius=0.06)
            dots_norm_1.add(dot)
            if i > 0:
                line = Line(
                    helper_axes.c2p(x_values[i-1], y_normed_1[i-1]),
                    helper_axes.c2p(x, y),
                    stroke_width=2,
                )
                lines_norm_1.add(line)
        dots_norm_1.set_color(YELLOW)
        lines_norm_1.set_color(YELLOW)

        dots_norm_2 = VGroup()
        lines_norm_2 = VGroup()
        for i, (x, y) in enumerate(zip(x_values, y_normed_2)):
            dot = Dot(helper_axes.c2p(x, y), radius=0.06)
            dots_norm_2.add(dot)
            if i > 0:
                line = Line(
                    helper_axes.c2p(x_values[i-1], y_normed_2[i-1]),
                    helper_axes.c2p(x, y),
                    stroke_width=2,
                )
                lines_norm_2.add(line)
        dots_norm_2.set_color(BLUE)
        lines_norm_2.set_color(BLUE)


        # self.remove(dots_norm_1)
        # self.remove(dots_norm_2)

        # self.add(axis_0_group_copy[1])
        self.wait()
        self.play(ReplacementTransform(axis_0_group_copy[1], dots_norm_1),
                  ReplacementTransform(axis_0_group_copy[2], lines_norm_1),
                  ReplacementTransform(dots_net_2_neuron_2, dots_norm_2),
                  ReplacementTransform(lines_net_2_neuron_2, lines_norm_2),
                  run_time=2)


        correlation_number=Tex('-0.77', font_size=46)
        correlation_number.set_color(MAGENTA)
        correlation_number.move_to([7.5, -0.15, 0 ])


        all_svgs[11].scale(1.9)
        all_svgs[11].move_to([7.5, -1.85, 0])

        all_svgs[12].scale(1.3)
        all_svgs[12].move_to([7.3, -1.1, 0])

        dots_norm_1_copy=dots_norm_1.copy()
        lines_norm_1_copy=lines_norm_1.copy()
        dots_norm_2_copy=dots_norm_2.copy()
        lines_norm_2_copy=lines_norm_2.copy()


        self.wait()
        self.play(ReplacementTransform(dots_norm_1_copy, correlation_number),
                  ReplacementTransform(lines_norm_1_copy, correlation_number),
                  ReplacementTransform(dots_norm_2_copy, correlation_number),
                  ReplacementTransform(lines_norm_2_copy, correlation_number),
                  dots_norm_1.animate.set_opacity(0.5),
                  lines_norm_1.animate.set_opacity(0.5),
                  dots_norm_2.animate.set_opacity(0.5),
                  lines_norm_2.animate.set_opacity(0.5),
                  self.frame.animate.reorient(0, 0, 0, (2.65, -0.2, 0.0), 7.81),
                  run_time=3)

        self.play(Write(all_svgs[11]),
                  Write(all_svgs[12]),
                  run_time=4)


        # self.add(all_svgs[11])
        # self.add(all_svgs[12])

        # self.add(embedding_network_2, embedding_network_1)

        M, grid_lines, diagonal_squares, off_diagonal_squares=make_correlation_matrix(N=20, noise_scale=0.8, center=[3, -1, 0], total_size=4.0, seed=42, line_stroke_width=1.0)

        # self.add(grid_lines, diagonal_squares, off_diagonal_squares)
        # self.remove(grid_lines, diagonal_squares, off_diagonal_squares)

        self.wait()
        self.remove(dots_norm_1, lines_norm_1, dots_norm_2, lines_norm_2,
                   axis_0_group_copy[0], axis_1_group_b[0])


        # self.remove(grid_lines)

        all_svgs[10].scale(0.83)
        all_svgs[10][:6].move_to([3.01, 1.25, 0])
        all_svgs[10][6:].move_to([0.75, -1, 0])


        # embedding_network_1.scale(0.75)
        # embedding_network_1.move_to([-0.15, -1.05, 0])
        # self.add(all_svgs[10][:6]) #Top 3 arrows
        # self.add(all_svgs[10][6:])
        # self.add(grid_lines)

        self.wait()
        self.play(embedding_network_1[1:].animate.scale(0.75).move_to([-0.15, -1.05, 0]),
                  ShowCreation(grid_lines), 
                  Write(all_svgs[10][6:]),
                  run_time=4)

        self.play(embedding_network_2[1:].animate.scale(0.75).flip(UP).rotate(90*DEGREES, [0, 0, 1]).move_to([3.05, 2.15, 0]),
                  Write(all_svgs[10][:6]),
                  self.frame.animate.reorient(0, 0, 0, (3.1, -0.15, 0.0), 6.72),
                  run_time=4
                  )

        # embedding_network_2[1:].scale(0.75)
        # embedding_network_2[1:].flip(UP)
        # embedding_network_2[1:].rotate(90*DEGREES, [0, 0, 1])
        # embedding_network_2[1:].move_to([3.05, 2.15, 0])

        # self.add(off_diagonal_squares)
        # self.add(diagonal_squares)
        # self.add(grid_lines)


        self.wait()
        self.play(ReplacementTransform(correlation_number, off_diagonal_squares[0]), 
                  FadeOut(all_svgs[12]),
                  FadeOut(all_svgs[11]),
                 run_time=3)
        self.play(FadeIn(off_diagonal_squares[1:]),
                  FadeIn(diagonal_squares),
                  run_time=3)
    

        self.wait()
        self.play(FadeOut(off_diagonal_squares), run_time=3)

        self.wait()
        self.play(FadeIn(off_diagonal_squares), 
                  FadeOut(diagonal_squares), run_time=3)



        M, grid_lines_2, diagonal_squares_2, off_diagonal_squares_2=make_correlation_matrix(N=20, noise_scale=0.0, center=[8, -1, 0], total_size=4.0, seed=42, line_stroke_width=1.0)

        all_svgs[13].scale(1.1)
        all_svgs[13].move_to([5.5, -3.2, 0])
        # self.add(grid_lines_2)
        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (4.93, -0.31, 0.0), 6.84),
                  ShowCreation(grid_lines_2),
                  ShowCreation(diagonal_squares_2),
                  FadeIn(diagonal_squares),
                  Write(all_svgs[13]),
                  run_time=4)

        all_svgs[7].scale(1.1)
        all_svgs[7].next_to(grid_lines_2, UP, buff=0.5)

        self.wait()
        self.play(Write(all_svgs[7]), run_time=5)

        all_svgs[6].move_to([5.3, 3.3, 0])

        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (5.2, 0.12, 0.0), 7.71),
                  Write(all_svgs[6]), 
                  run_time=4)


        # self.add(all_svgs[6]) #Title
        # self.add(all_svgs[7])




        self.wait(20)
        self.embed()





















