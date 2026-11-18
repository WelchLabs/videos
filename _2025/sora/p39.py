from welch_paths import welch_path
from pathlib import Path

import manimgx as gx
from welch_geometry import diagram_axes
from welch_scene import WelchScene, image_at_height
from welch_text import math_tex

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'



class P39v3(WelchScene):
    def construct(self):
        '''
        Critical thing at the end here will be a smooth transition to the axes and positioning of p40, 
        I should be able to smoothly zoom out and draw a spiral. 

        '''
        axes = gx.SVGMobject(welch_path('sora/graphics/to_manim/p39_2.svg'))[1:]
        axes.move_to(gx.ORIGIN).scale(4).shift(gx.LEFT * 3)
        x1 = gx.VGroup(axes[6], axes[7])
        x2 = gx.VGroup(axes[8], axes[9])
        x3 = gx.VGroup(axes[10], axes[11])
        xn = gx.VGroup(axes[15], axes[16])
        
        x1_tick = axes[12]
        x2_tick = axes[13]
        x3_tick = axes[14]
        
        dots = gx.VGroup(axes[19], axes[20], axes[21])
        
        x1_arrow = gx.VGroup(axes[2], axes[3])
        x2_arrow = gx.VGroup(axes[0], axes[1])
        x3_arrow = gx.VGroup(axes[4], axes[5])
        xn_arrow = gx.VGroup(axes[17], axes[18])
        
        final_point = axes[22]
        axes_no_ticks=gx.VGroup(x1_arrow, x1, x2_arrow, x2, x3_arrow, x3, xn_arrow, xn, dots)

        img = Image.open('/Users/stephen/manim/videos/_2025/sora/dog_128.jpg')
        img_array = np.array(img) #Let's try 64x64
        pixel_dimensions = img_array.shape  
        height, width = img_array.shape[:2]
        pixel_squares = gx.VGroup()
        pixel_size = 3.0 / height
        
        for i in range(height):
            for j in range(width):
                if len(img_array.shape) == 3: 
                    r, g, b = img_array[i, j][:3]
                    color = gx.rgb_to_color([r/255, g/255, b/255])
                else: 
                    gray = img_array[i, j]
                    color = gx.rgb_to_color([gray/255, gray/255, gray/255])
                
                pixel_square = gx.Square(side_length=pixel_size, fill_color=color, fill_opacity=1.0, stroke_width=0)
                
                x_pos = (j - width/2 + 0.5) * pixel_size
                y_pos = -(i - height/2 + 0.5) * pixel_size
                
                pixel_square.move_to([x_pos, y_pos, 0])
                pixel_squares.add(pixel_square)
        
    
        self.wait()
        self.play(gx.FadeIn(pixel_squares))
        self.wait()

        # pixel_squares.move_to([3.5, 0, 0])
        self.play(pixel_squares.animate.move_to([3.0, 0, 0]), 
                  gx.Create(axes_no_ticks),
                  run_time=3.0)
        self.wait()

        self.play(gx.ReplacementTransform(pixel_squares.copy(), final_point), run_time=2.5) #Magenta circle mid way through?
        self.wait()

        #Ok now big zoom in and dilattion -> yeah actually do I really need ot dilate? I mean ig might be cool
        # But yeah now sure if it's needed???


        # self.add(r2, l2)

        # self.remove(pixel_squares[0])
        # self.add(pixel_squares[0])

        ## Hey what about zooming way in on the upper left corner of the image while I do the dilation???
        ## That might help clarify things nicely

        ul_corner=pixel_squares.get_corner(gx.UL)
        self.play(pixel_squares.animate.scale(15).move_to(ul_corner, aligned_edge=gx.LEFT+gx.UP), run_time=4.0)
        self.wait()

        x1_tick.shift([-0.5, 0, 0]) #Drawing is a bit off
        r1=gx.SurroundingRectangle(pixel_squares[0], buff=0.0)
        r1.set_stroke(width=5, color='#00FFFF', opacity=1.0)
        l1=gx.Line(r1.get_corner(gx.DL), x1_tick.get_top())
        # l1.set_opacity(1.0)
        l1.set_stroke(width=5, color='#00FFFF')

        x2_tick.shift([0, 0.07, 0]) #Drawing is a bit off
        r2=gx.SurroundingRectangle(pixel_squares[1], buff=0.0)
        r2.set_stroke(width=5, color='#FF00FF', opacity=1.0)
        l2=gx.Line(r2.get_corner(gx.DL), x2_tick.get_right())
        l2.set_stroke(width=5, color='#FF00FF')


        self.play(gx.Create(r1))
        self.play(gx.Create(l1))
        self.play(gx.Create(x1_tick))
        self.wait()

        self.play(gx.Create(r2))
        self.play(gx.Create(l2))
        self.play(gx.Create(x2_tick))
        self.wait()

        # "If we reduce our image to only two pixels"
        # Get rid of everything execept to pixels and cyan and magent boxes
        # move sligtly apart
        # Put top center
        # Bring in same or very similiar axes to p40 so can make the transition smoooooooth. 

        pixel_one_group=gx.VGroup(pixel_squares[0], r1)
        pixel_two_group=gx.VGroup(pixel_squares[1], r2)

        self.play(pixel_one_group.animate.move_to([5-.01, 1.8, 0], aligned_edge=gx.RIGHT), #.scale(2.0),
                  pixel_two_group.animate.move_to([5.01, 1.8, 0], aligned_edge=gx.LEFT), #.scale(2.0),
                  gx.FadeOut(axes),
                  gx.FadeOut(pixel_squares[2:]),
                  gx.FadeOut(l1),
                  gx.FadeOut(l2),
                  self.animate_view(0, 0, 0, (3.39, 1.65, 0.0), 4.31),
                  run_time=4.0)
        # self.wait()

        pixel_one_group.move_to([5-.015, 1.8, 0], aligned_edge=gx.RIGHT)
        pixel_two_group.move_to([5.015, 1.8, 0], aligned_edge=gx.LEFT)

        #Now bring in p40 style axis. 

        axes = diagram_axes(
            x_range=[-1.2, 1.2, 0.5],
            y_range=[-1.2, 1.2, 0.5],
            height=7,
            width=7,
            axis_config={
                "color": CHILL_BROWN, 
                "stroke_width": 2,
                "include_tip": True,
                "include_ticks": False, #I don't think i want ticks here - transition to - actually do I want ticks like ever? The spacing is weird for some reason...
                "tick_size": 0.06,
                "tip_config": {"color": CHILL_BROWN, "length": 0.15, "width": 0.15}
            }
        )

        # Hmm manually add ticks and labels here?
        # I think that's probably the move...
        x_label=math_tex('x', font_size=28)
        x_label.set_color(CHILL_BROWN)
        x_label.move_to(axes.get_right()).shift([-0.08, -0.3, 0])

        y_label=math_tex('y', font_size=28)
        y_label.set_color(CHILL_BROWN)
        y_label.move_to(axes.get_top()).shift([-0.28, -0.08, 0])

        tick_x=gx.Line([3.0, -0.07, 0], [3.0, -0.18, 0])
        tick_x.set_stroke(width=2, color=CHILL_BROWN)
        tick_label_x=math_tex('1', font_size=20)
        tick_label_x.set_color(CHILL_BROWN)
        tick_label_x.next_to(tick_x, gx.DOWN, buff=0.05)

        tick_y=gx.Line([-0.07, 3.0, 0], [-0.18, 3.0, 0])
        tick_y.set_stroke(width=2, color=CHILL_BROWN)
        tick_label_y=math_tex('1', font_size=20)
        tick_label_y.set_color(CHILL_BROWN)
        tick_label_y.next_to(tick_y, gx.LEFT, buff=0.05)

        axis_accessories=gx.VGroup(x_label, y_label, tick_x, tick_label_x, tick_y, tick_label_y)

        self.wait()
        self.play(gx.FadeIn(axes), gx.FadeIn(axis_accessories))
        self.wait()

        #Ok i think just move and shink each pixel to the end of each axis? 

        self.play(pixel_one_group.animate.scale(0.5).move_to(axes.get_right()).shift([0.05, 0.12, 0]), run_time=2.0)
        self.wait()

        self.play(pixel_two_group.animate.scale(0.5).move_to(axes.get_top()).shift([0.15, 0.00, 0]), run_time=2.0)
        self.wait()

        # pixel_one_group.scale(0.5)
        # pixel_one_group.move_to(axes.get_right()).shift([0.05, 0.12, 0])
        # pixel_one_group.move_to(x_label)
        # self.remove(pixel_one_group[0])

        #Ok ok now for the points themselves, I think I just do them quickly in illustrator
        # pixel_two_group.move_to(axes.get_top()).shift([0.15, 0.00, 0])

        #Self.wait

        self.play(self.animate_view(0,0,0,(0,0,0), 8),
                gx.FadeOut(axis_accessories),
                gx.FadeOut(pixel_one_group),
                gx.FadeOut(pixel_two_group),
                run_time=4.0) #Match p40 framing. 

        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class P39v2(WelchScene):
    def construct(self):
        '''
        Critical thing at the end here will be a smooth transition to the axes and positioning of p40, 
        I should be able to smoothly zoom out and draw a spiral. 

        '''

        cat = image_at_height('/Users/stephen/manim/videos/_2025/sora/n02123045_1955.jpg').scale(1)
        # cat.move_to([-3.5, 0, 0])


        axes = gx.SVGMobject(welch_path('sora/graphics/to_manim/p39_2.svg'))[1:]
        axes.move_to(gx.ORIGIN).scale(4).shift(gx.LEFT * 3)
        
        
        x1 = gx.VGroup(axes[6], axes[7])
        x2 = gx.VGroup(axes[8], axes[9])
        x3 = gx.VGroup(axes[10], axes[11])
        xn = gx.VGroup(axes[15], axes[16])
        
        x1_tick = axes[12]
        x2_tick = axes[13]
        x3_tick = axes[14]
        
        dots = gx.VGroup(axes[19], axes[20], axes[21])
        
        x1_arrow = gx.VGroup(axes[2], axes[3])
        x2_arrow = gx.VGroup(axes[0], axes[1])
        x3_arrow = gx.VGroup(axes[4], axes[5])
        xn_arrow = gx.VGroup(axes[17], axes[18])
        
        final_point = axes[22]


        axes_no_ticks=gx.VGroup(x1_arrow, x1, x2_arrow, x2, x3_arrow, x3, xn_arrow, xn, dots)


        self.wait()
        self.play(gx.FadeIn(cat))
        self.wait()

        self.play(cat.animate.move_to([3.5, 0, 0]), 
                  gx.Create(axes_no_ticks),
                  run_time=3.0)
        self.add(final_point)


        # self.play(Transform(cat, final_point))


        # self.play(ShowCreation(axes_no_ticks), run_time=2.5)
        self.wait()

        #Ok let me consider the "expanding the image into pixels deal here - I think that could work pretty well?"
        img = Image.open('/Users/stephen/manim/videos/_2025/sora/cat_downscaled.jpg')
        img_array = np.array(img) #Let's try 64x64
        pixel_dimensions = img_array.shape  
        height, width = img_array.shape[:2]
        pixel_squares = gx.VGroup()
        pixel_size = cat.height / height
        
        for i in range(height):
            for j in range(width):
                if len(img_array.shape) == 3: 
                    r, g, b = img_array[i, j][:3]
                    color = gx.rgb_to_color([r/255, g/255, b/255])
                else: 
                    gray = img_array[i, j]
                    color = gx.rgb_to_color([gray/255, gray/255, gray/255])
                
                pixel_square = gx.Square(side_length=pixel_size, fill_color=color, fill_opacity=1.0, stroke_width=0)
                
                x_pos = (j - width/2 + 0.5) * pixel_size
                y_pos = -(i - height/2 + 0.5) * pixel_size
                
                pixel_square.move_to([x_pos, y_pos, 0])
                pixel_squares.add(pixel_square)
        
        pixel_squares.move_to(cat.get_center())


        # I wonder if fading in while dilating would be cool?
        self.add(pixel_squares)

        animations = []
        center = pixel_squares.get_center()
        gap_factor = 0.4  
        
        for pixel in pixel_squares:
            pixel_pos = pixel.get_center()
            direction_vector = pixel_pos - center
            
            distance = np.linalg.norm(direction_vector)
            
            if distance > 0:
                unit_vector = direction_vector / distance
                
                displacement = unit_vector * distance * gap_factor
                new_position = pixel_pos + displacement
                
                animations.append(gx.ApplyMethod(pixel.move_to, new_position))
                
        new_pixel = gx.VGroup()
        for pixel in pixel_squares:
            new_pixel.add(pixel)

        self.play(*animations, run_time=2)


        #Maybe when we go down to 2 pixels I switch out the axes in preparation for the move to 40



        # self.remove(axes)

        self.wait(20)
        # Scene authoring stops here in the interactive editor.

class P39v1(WelchScene):
    def construct(self):
        cat = image_at_height(HERE / 'n02123045_1955.jpg').scale(1)
        self.play(gx.FadeIn(cat))
        self.wait(1)
        
        img = Image.open(HERE / 'cat_downscaled.jpg')
        img_array = np.array(img)
        num_pixels = (32, 32) 
        
        img_resized = img.resize(num_pixels, Image.NEAREST)
        img_array = np.array(img_resized)
        pixel_dimensions = img_array.shape  
        height, width = img_array.shape[:2]
        
        pixel_squares = gx.VGroup()
        
        pixel_size = cat.height / height
        
        for i in range(height):
            for j in range(width):
                if len(img_array.shape) == 3: 
                    r, g, b = img_array[i, j][:3]
                    color = gx.rgb_to_color([r/255, g/255, b/255])
                else: 
                    gray = img_array[i, j]
                    color = gx.rgb_to_color([gray/255, gray/255, gray/255])
                
                pixel_square = gx.Square(side_length=pixel_size, fill_color=color, fill_opacity=1.0, stroke_width=0)
                
                x_pos = (j - width/2 + 0.5) * pixel_size
                y_pos = -(i - height/2 + 0.5) * pixel_size
                
                pixel_square.move_to([x_pos, y_pos, 0])
                pixel_squares.add(pixel_square)
        
        pixel_squares.move_to(cat.get_center())
        
        
        
        self.play(gx.FadeOut(cat))
        
        
        self.play(gx.FadeIn(pixel_squares))
        
        
        animations = []
        center = pixel_squares.get_center()
        gap_factor = 0.4  
        
        for pixel in pixel_squares:
            pixel_pos = pixel.get_center()
            direction_vector = pixel_pos - center
            
            distance = np.linalg.norm(direction_vector)
            
            if distance > 0:
                unit_vector = direction_vector / distance
                
                displacement = unit_vector * distance * gap_factor
                new_position = pixel_pos + displacement
                
                animations.append(gx.ApplyMethod(pixel.move_to, new_position))
                
        new_pixel = gx.VGroup()
        for pixel in pixel_squares:
            new_pixel.add(pixel)
            
                
        
        

        self.play(*animations, run_time=2)
        
        down = gx.VGroup(pixel_squares[992], pixel_squares[1023])
        right = gx.VGroup(pixel_squares[0], pixel_squares[1023])
        
        down_bracket = gx.Brace(down, direction=gx.DOWN)
        down_bracket_text = math_tex("32 px").next_to(down_bracket, gx.DOWN)
        right_bracket = gx.Brace(right, direction=gx.RIGHT)
        right_bracket_text = math_tex("32 px").next_to(right_bracket, gx.RIGHT)
        
        self.add(down_bracket, down_bracket_text, right_bracket, right_bracket_text)
        
        self.wait(2)
        
        self.remove(down_bracket, down_bracket_text, right_bracket, right_bracket_text)
        
        self.wait()
        
        self.play(pixel_squares.animate.shift(gx.LEFT * 3.5))
        
        axes = gx.SVGMobject(HERE / 'p39_2.svg')[1:]
        axes.move_to(gx.ORIGIN).scale(4).shift(gx.RIGHT * 3)
        
        
        x1 = gx.VGroup(axes[6], axes[7])
        x2 = gx.VGroup(axes[8], axes[9])
        x3 = gx.VGroup(axes[10], axes[11])
        xn = gx.VGroup(axes[15], axes[16])
        
        x1_tick = axes[12]
        x2_tick = axes[13]
        x3_tick = axes[14]
        
        dots = gx.VGroup(axes[19], axes[20], axes[21])
        
        x1_arrow = gx.VGroup(axes[2], axes[3])
        x2_arrow = gx.VGroup(axes[0], axes[1])
        x3_arrow = gx.VGroup(axes[4], axes[5])
        xn_arrow = gx.VGroup(axes[17], axes[18])
        
        final_point = axes[22]
        
        self.wait()
        
        self.play(gx.FadeIn(x1_arrow), gx.FadeIn(x2_arrow), gx.FadeIn(x3_arrow), gx.FadeIn(x1), gx.FadeIn(x2), gx.FadeIn(x3))
        
        self.wait()
        
        pixel_squares[0].set_stroke(gx.WHITE, width=4)
        
        self.play(pixel_squares[0].animate.scale(3).next_to(x1_tick, gx.DOWN), run_time = 1)
        self.play(
            gx.LaggedStart(
            gx.Indicate(x1_arrow),
            gx.ReplacementTransform(pixel_squares[0], x1_tick),
            lag_ratio=0.75
            ),
            run_time=2
        )
        
        self.wait()
        
        pixel_squares[1].set_stroke(gx.WHITE, width=4)
        
        self.play(pixel_squares[1].animate.scale(3).next_to(x2_tick, gx.LEFT), run_time = 1)
        self.play(
            gx.LaggedStart(
            gx.Indicate(x2_arrow),
            gx.ReplacementTransform(pixel_squares[1], x2_tick),
            lag_ratio=0.75
            ),
            run_time=2
        )
        
        self.wait()
        
        pixel_squares[2].set_stroke(gx.WHITE, width=4)
        self.play(pixel_squares[2].animate.scale(3).next_to(x3_tick, gx.DOWN), run_time = 1)
        self.play(
            gx.LaggedStart(
            gx.Indicate(x3_arrow),
            gx.ReplacementTransform(pixel_squares[2], x3_tick),
            lag_ratio=0.75
            ),
            run_time=2
        )
        
        self.wait()
        
        other_pixels = gx.VGroup(*[pixel_squares[i] for i in range(3, len(pixel_squares))])
        xn_group = gx.VGroup(xn_arrow, xn, dots)
        self.play(
            gx.LaggedStart(gx.ReplacementTransform(other_pixels, final_point), gx.FadeIn(xn_group), lag_ratio=0.75)
            , run_time = 3
        )
        
        self.play(axes.animate.move_to(gx.ORIGIN))


        

        



        # Scene authoring stops here in the interactive editor.

def label_with_indices(self, mobject, scale=0.4, color=YELLOW):
    for i, submob in enumerate(mobject.submobjects):
        label = gx.Text(str(i)).scale(scale).set_color(color)
        label.move_to(submob.get_center())
        self.add(label)