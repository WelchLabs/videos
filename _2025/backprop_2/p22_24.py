from welch_paths import welch_path
import manimgx as m
from welch_scene import WelchScene
from welch_text import math_tex
from functools import partial
import numpy as np
import torch
import glob

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'
GREEN='#00a14b'

svg_path=welch_path('backprop2/graphics/to_manim')
data_path=welch_path('backprop2/hackin')


def format_number(num, total_chars=6, align='right'):
    """
    Format number to maintain consistent visual alignment for animations.
    
    Args:
        num: The number to format
        total_chars: Total character width (should accommodate largest expected number)
        align: 'right', 'left', or 'center' - how to align within the fixed width
    """
    abs_num = abs(num)
    
    # Determine appropriate precision based on magnitude
    if abs_num >= 100:
        # 100+: no decimal places (e.g., "123", "-123")
        formatted = f"{num:.0f}"
    elif abs_num >= 10:
        # 10-99: one decimal place (e.g., "12.3", "-12.3")  
        formatted = f"{num:.1f}"
    elif abs_num >= 1:
        # 1-9: two decimal places (e.g., "1.23", "-1.23")
        formatted = f"{num:.2f}"
    else:
        # Less than 1: two decimal places (e.g., "0.12", "-0.12")
        formatted = f"{num:.2f}"
    
    # Pad to consistent width
    if align == 'right':
        return formatted.rjust(total_chars)
    elif align == 'left':
        return formatted.ljust(total_chars)
    else:  # center
        return formatted.center(total_chars)

def format_number_fixed_decimal(num, decimal_places=2, total_chars=6):
    """
    Alternative formatter that keeps decimal point in same position.
    Useful when you want all numbers to have the same decimal precision.
    """
    formatted = f"{num:.{decimal_places}f}"
    return formatted.rjust(total_chars)

def get_numbers_2(x, w, logits, yhats):

    numbers = m.VGroup()

    tx = math_tex(str(x) + r'^\circ')
    tx.scale(0.13)
    tx.move_to([-1.49, 0.02, 0])
    numbers.add(tx)
    
    # Weights - using consistent formatting
    tm1 = math_tex(format_number(w[0], total_chars=6)).set_color('#00FFFF')
    tm1.scale(0.16)
    tm1.move_to([-1.195, 0.205, 0])
    numbers.add(tm1)
    
    tm2 = math_tex(format_number(w[1], total_chars=6)).set_color(YELLOW)
    tm2.scale(0.15)
    tm2.move_to([-1.155, 0.015, 0])
    numbers.add(tm2)
    
    tm3 = math_tex(format_number(w[2], total_chars=6)).set_color(GREEN)
    tm3.scale(0.16)
    tm3.move_to([-1.19, -0.17, 0])
    numbers.add(tm3)
    
    # Biases
    tb1 = math_tex(format_number(w[3], total_chars=6)).set_color('#00FFFF')
    tb1.scale(0.16)
    tb1.move_to([-0.875, 0.365, 0])
    numbers.add(tb1)
    
    tb2 = math_tex(format_number(w[4], total_chars=6)).set_color(YELLOW)
    tb2.scale(0.16)
    tb2.move_to([-0.875, 0.015, 0])
    numbers.add(tb2)
    
    tb3 = math_tex(format_number(w[5], total_chars=6)).set_color(GREEN)
    tb3.scale(0.16)
    tb3.move_to([-0.88, -0.335, 0])
    numbers.add(tb3)
    
    # Logits
    tl1 = math_tex(format_number(logits[0], total_chars=6)).set_color('#00FFFF')
    tl1.scale(0.16)
    tl1.move_to([-0.52, 0.37, 0])
    numbers.add(tl1)
    
    tl2 = math_tex(format_number(logits[1], total_chars=6)).set_color(YELLOW)
    tl2.scale(0.16)
    tl2.move_to([-0.52, 0.015, 0])
    numbers.add(tl2)
    
    tl3 = math_tex(format_number(logits[2], total_chars=6)).set_color(GREEN)
    tl3.scale(0.16)  
    tl3.move_to([-0.52, -0.335, 0])
    numbers.add(tl3)
    
    # Predictions
    yhat1 = math_tex(f"{yhats[0]:.3f}").set_color('#00FFFF')
    yhat1.scale(0.16)
    yhat1.move_to([0.22, 0.37, 0])
    numbers.add(yhat1)
    
    yhat2 = math_tex(f"{yhats[1]:.3f}").set_color(YELLOW)
    yhat2.scale(0.16)
    yhat2.move_to([0.22, 0.015, 0])
    numbers.add(yhat2)
    
    yhat3 = math_tex(f"{yhats[2]:.3f}").set_color(GREEN)
    yhat3.scale(0.16)
    yhat3.move_to([0.22, -0.335, 0])
    numbers.add(yhat3)
    
    return numbers



class p22_24(WelchScene):
    def construct(self):
        '''
        
        '''
        net_background=m.SVGMobject(svg_path+'/p44_background_1.svg')

        self.add(net_background)
        # self.set_view(0, 0, 0, (-0.07, -0.02, 0.0), 1.91)
        # self.set_view(0, 0, 0, (-0.22, -0.03, 0.0), 1.74)
        # self.set_view(0, 0, 0, (-0.31, -0.02, 0.0), 1.62)
        self.set_view(0, 0, 0, (-0.56, 0.23, 0.0), 1.56)

        data=np.load(data_path+'/cities_1d_2.npy')
        xs=data[:,:2]
        ys=data[:,2]
        weights=data[:,3:9]
        grads=data[:,9:15]
        logits=data[:,15:18]
        yhats=data[:, 18:]

        x=2.3514
        w=[1, 0, -1, 0, 0, 0]
        logits=[2.35, 0, -2.35]
        yhats=[0.905, 0.086, 0.008]
        nums=get_numbers_2(x, w, logits, yhats)
        self.add(nums)

        net_background.shift([0,0.23,0])
        nums.shift([0,.23,0])

        # self.wait()

        layers=m.VGroup()
        for p in sorted(glob.glob(svg_path+'/p22_24/*.svg')):
            layers.add(m.SVGMobject(p)[1:])
         
        yhat2 = math_tex("0.086").set_color(YELLOW)
        yhat2.scale(0.14)
        yhat2.move_to([1.205, 0.18, 0])
        loss = math_tex(format_number(2.45, total_chars=6)).set_color(YELLOW)
        loss.scale(0.16)
        loss.move_to([1.5, 0.18, 0])

        self.wait()
        self.play(self.animate_view(0, 0, 0, (0.0, 0.07, 0.0), 1.97),
                  m.FadeIn(layers[:2]), m.FadeIn(yhat2), m.FadeIn(loss), run_time=3)

        self.wait()

        self.play(m.Create(layers[3]))

        m2_label=net_background[42:44].copy()
        loss_label=layers[1][:4].copy()
        self.add(m2_label, loss_label)

        self.play(loss_label.animate.scale(0.75).move_to([0.7, 0.05, 0]), run_time=1.0)
        self.play(m2_label.animate.scale(1.2).move_to([1.48, -0.67, 0]), run_time=4.0)

        self.wait()

        m0=nums[2].copy()
        loss_copy = loss.copy()
        self.play(m0.animate.move_to([0.79, -0.68, 0]), run_time=3.0)
        self.play(loss_copy.animate.move_to([0.7, -0.05, 0]), run_time=1.2)
        self.add(layers[2])

        self.wait()

        box = m.SurroundingRectangle(nums[2], color=YELLOW, buff=0.025)
        self.play(m.Create(box))
        tm2 = math_tex(format_number(0.1, total_chars=6)).set_color(YELLOW)
        tm2.scale(0.15)
        tm2.move_to([-1.155, 0.015+0.23, 0])
        self.play(m.Transform(nums[2], tm2))
        self.wait()

        #Ok, now update all yhs and yhats at once!
        w=[1, 0.1, -1, 0, 0, 0]
        logits=[2.35, 0.24, -2.35]
        yhats=[0.885, 0.107, 0.008]
        nums_2=get_numbers_2(x, w, logits, yhats)
        nums_2.shift([0,.23,0])

        self.play(*[m.Transform(nums[i], nums_2[i]) for i in [-6, -5, -4, -3, -2, -1]])
        self.wait()

        yhat2_new = nums_2[-2].copy()
        loss_new = math_tex(format_number(2.24, total_chars=6)).set_color(YELLOW)
        loss_new.scale(0.16)
        loss_new.move_to([1.5, 0.18, 0])
        self.play(yhat2_new.animate.scale(0.9).move_to([1.205, 0.18, 0]), 
                  m.FadeOut(yhat2), run_time=2.0)
        self.play(m.Transform(loss, loss_new))
        self.wait()

        m1=nums_2[2].copy()
        loss_copy_2 = loss_new.copy()
        self.play(m1.animate.move_to([1.25, -0.68, 0]), run_time=2.0)
        self.play(loss_copy_2.animate.move_to([0.7, -0.49, 0]), run_time=1.0)  
        self.add(layers[4])
        self.wait()

        self.play(m.FadeIn(layers[5]))
        self.wait()
        self.play(m.FadeIn(layers[6]))
        self.wait()

        net=m.VGroup(net_background, nums)
        plot_to_move=m.VGroup(layers[2:5], m2_label, loss_label)

        self.wait()
        self.remove(box); nums[-6:].set_color(CHILL_BROWN); nums[0].set_color(CHILL_BROWN)
        self.play(m.FadeOut(layers[0:2]), m.FadeOut(layers[5:7]),
                  m.FadeOut(loss_copy_2), m.FadeOut(m1), m.FadeOut(m0), m.FadeOut(loss_copy),
                  m.FadeOut(loss_new), m.FadeOut(loss), m.FadeOut(yhat2_new),
                  net.animate.scale(0.78).move_to([0.38,0.018,0]),
                  plot_to_move.animate.scale(0.58).move_to([-0.68, 0.645, 0]),
                  self.animate_view(0, 0, 0, (-0.05, -0.02, 0.0), 2.00),
                  run_time=3.0)
        self.add(layers[7])
        self.add(layers[8])
        self.remove(plot_to_move)
        self.wait()

        self.play(m.Create(layers[9]))
        self.wait()

        m2_plot=layers[8][-21:]
        # self.add(m2_plot_copy)
        # to_fade=VGroup(layers[8][:-21], layers[7], layers[9], net)

        self.play(m.FadeOut(layers[8][:-21]), m.FadeOut(layers[7]), m.FadeOut(layers[9]), m.FadeOut(net))
        self.play(m2_plot.animate.scale(2.2).move_to([-0.19,-0.015,0]), run_time=1.5)
        self.add(layers[10])
        self.wait()

        # self.add(layers[9])

        
        # net.scale(0.78)
        # net.move_to([0.38,0.018,0])

        
        # plot_to_move.scale(0.58)
        # plot_to_move.move_to([-0.68, 0.645, 0])

        # self.set_view(0, 0, 0, (-0.05, -0.02, 0.0), 2.00)

        # # nums_2[-6:].set_color(CHILL_BROWN)
        # nums[-6:].set_color(CHILL_BROWN)

        # yhat2_new.scale(0.14)
        # yhat2_new.move_to([1.205, 0.18, 0])




        # m3_label.scale(1.2)
        # m3_label.move_to([1.48, -0.67, 0])
        # self.add(m3_label)

        
        # loss_label.scale(0.75)
        # loss_label.move_to([[0.7, 0.05, 0]])
        # self.add(loss_label)




        # loss_copy = loss.copy()
        # loss_copy.move_to([0.7, -0.05, 0])
        # self.add(loss_copy)

 


        #DON'T FORGET TO UPDATE ALL PROBS when we change the one wieght!




        # box = SurroundingRectangle(nums[2], color=YELLOW, buff=0.025)
        # self.play(ShowCreation(box))
        # self.wait()




















        self.wait()
        pass
