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
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'

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

def get_numbers_3(x, w, logits, yhats):

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
    yhat1 = math_tex(format_number(yhats[0], total_chars=6)).set_color('#00FFFF')
    yhat1.scale(0.16)
    yhat1.move_to([0.22, 0.37, 0])
    numbers.add(yhat1)
    
    yhat2 = math_tex(format_number(yhats[1], total_chars=6)).set_color(YELLOW)
    yhat2.scale(0.16)
    yhat2.move_to([0.22, 0.015, 0])
    numbers.add(yhat2)
    
    yhat3 = math_tex(format_number(yhats[2], total_chars=6)).set_color(GREEN)
    yhat3.scale(0.16)
    yhat3.move_to([0.22, -0.335, 0])
    numbers.add(yhat3)
    
    
    return numbers


class p32_40_v2(WelchScene):
    def construct(self):
        '''
        
        '''
        net_background=m.SVGMobject(svg_path+'/p44_background_1.svg')[1:]

        x=2.3514
        w=[1, 0, -1, 0, 0, 0]
        logits=[2.35, 0, -2.35]
        yhats=[0.91, 0.09, 0.00]
        nums=get_numbers_3(x, w, logits, yhats)
        
        rect_1=m.RoundedRectangle(width=0.65, height=1.1, corner_radius=0.02)
        rect_1.set_stroke(width=0)
        rect_1.set_color(CHILL_GREEN)
        rect_1.set_opacity(0.4)
        rect_1.move_to([-1.0, 0, 0])

        rect_2=m.RoundedRectangle(width=0.54, height=1.1, corner_radius=0.02)
        rect_2.set_stroke(width=0)
        rect_2.set_color(CHILL_BLUE)
        rect_2.set_opacity(0.4)
        rect_2.move_to([-0.15, 0.0, 0])

        dldm=math_tex(r"\frac{\partial L}{\partial m}")
        modular_eq_1=math_tex("=")
        modular_eq_2=math_tex(r"\frac{\partial h}{\partial m}")
        modular_eq_3=math_tex(r"\cdot")
        modular_eq_4=math_tex(r"\frac{\partial L}{\partial h}")

        modular_eq=m.VGroup(dldm, modular_eq_1, modular_eq_2, modular_eq_3, modular_eq_4)
        dldm.scale(0.17)
        modular_eq_1.scale(0.22)
        modular_eq_2.scale(0.17)
        modular_eq_3.scale(0.35)
        modular_eq_4.scale(0.17)
        dldm.move_to([-1.5, -0.7, 0])
        modular_eq_1.next_to(dldm, m.RIGHT, buff=0.18)
        modular_eq_2.next_to(modular_eq_1, m.RIGHT, buff=0.15)
        modular_eq_3.next_to(modular_eq_2, m.RIGHT, buff=0.4)
        modular_eq_4.next_to(modular_eq_3, m.RIGHT, buff=0.35)


        #p32c animation
        self.set_view(0, 0, 0, (-0.62, -0.1, 0.0), 1.56)
        self.add(net_background, nums)
        self.wait(0)
        self.play(m.FadeIn(dldm))
        self.play(m.FadeIn(rect_1), m.ReplacementTransform(dldm[3:].copy(), modular_eq_2[2:]))
        self.add(modular_eq_2, modular_eq_1)
        self.wait()
        self.play(m.FadeIn(rect_2), m.ReplacementTransform(dldm[:2].copy(), modular_eq_4[:2]))
        self.add(modular_eq_3, modular_eq_4)
        self.wait()


        #p33 animation
        # softmax_box=VGroup(net_background[18:34], net_background[11])
        # not_softmax_box=VGroup(net_background[:11], net_background[12:18], net_background[34:])
        softmax_box=m.VGroup(net_background[17:33], net_background[10])
        not_softmax_box=m.VGroup(net_background[:10], net_background[11:17], net_background[33:])

        # self.remove(not_softmax_box)
        softmax_derivative=m.SVGMobject(svg_path+'/p33_to_manim.svg')[1:]
        softmax_derivative.scale(1.17)
        softmax_derivative.move_to([1.4,-0.07,0])

        dldh_eq=math_tex(r"\frac{\partial L}{\partial h}=\hat{y}-y")
        dldh_eq.scale(0.25)
        dldh_eq.move_to([1.28, -0.95, 0])
        dldh_border=m.RoundedRectangle(width=dldh_eq.width + 0.1, height=dldh_eq.height + 0.1, corner_radius=0.02).move_to(dldh_eq.get_center())
        dldh_border.set_color(YELLOW).set_stroke(width=2)
        dldh_with_border=m.VGroup(dldh_eq, dldh_border)

        #p33a animation
        self.wait()
        # self.remove(rect_1, rect_2, not_softmax_box)
        self.play(nums.animate.set_opacity(0.0), modular_eq[:-1].animate.set_opacity(0.0),
                 rect_1.animate.set_opacity(0.0), rect_2.animate.set_opacity(0.0), not_softmax_box.animate.set_opacity(0.0))
        self.play(m.Create(softmax_derivative),
                  # self.animate_view(0, 0, 0, (0.86, -0.1, 0.0), 2.95),
                  self.animate_view(0, 0, 0, (1.35, -0.11, 0.0), 2.40),
                  m.ReplacementTransform(modular_eq[-1], dldh_eq[:5]),
                  run_time=3
                  )
        self.add(dldh_eq)
        self.wait()
        self.play(m.Create(dldh_border))
        self.wait()

        #p34 setup
        p34_blocks=m.VGroup()
        for p in sorted(glob.glob(svg_path+'/p34/*.svg')):
            p34_blocks.add(m.SVGMobject(p)[1:])
        p34_blocks[0].move_to([-0.02, 0.01, 0])
        p34_blocks[1].move_to([0.62, 0.54, 0])
        p34_blocks[2].move_to([-0.057, 0.047, 0])
        p34_blocks[4].move_to([1.4, 0.42, 0 ])
        p34_blocks[5].move_to([1.4, 0.0, 0 ])

        # Go ahead and load up modular net background we need for p35
        p35_net_background=m.VGroup()
        for p in sorted(glob.glob(svg_path+'/p_35/*.svg')):
            p35_net_background.add(m.SVGMobject(p)[1:])
        

        # make some backround vars actually vars for later
        x_label=math_tex('x').set_color(CHILL_BROWN).scale(0.16)
        x_label.move_to([-1.49, -0.05, 0])
        m2_label=math_tex('m_2').set_color(CHILL_BROWN).scale(0.11)
        m2_label.move_to([-1.155, -0.05, 0])
        b2_label=math_tex('b_2').set_color(CHILL_BROWN).scale(0.11)
        b2_label.move_to([-0.87, -0.055, 0])
        h2_label=math_tex('h_2').set_color(CHILL_BROWN).scale(0.11)
        h2_label.move_to([-0.51, -0.055, 0])
        manual_background_labels=m.VGroup(x_label, m2_label, b2_label, h2_label)
        self.wait()

        self.play(m.FadeOut(softmax_derivative), run_time=1.0)
        self.play(m.FadeIn(p35_net_background[:2]),
                  m.FadeIn(manual_background_labels),
                  # not_softmax_box.animate.set_opacity(1.0), 
                  nums.animate.set_opacity(1.0), 
                  dldh_with_border.animate.scale(0.8).move_to([1.28, 0.7, 0]),
                  self.animate_view(0, 0, 0, (-0.02, 0.0, 0.0), 1.90), run_time=2.0)
        self.remove(softmax_box)
        self.add(p34_blocks[0], p34_blocks[1][:-1])
        self.wait()

        y1=math_tex("0").set_color("#00FFFF").scale(0.19).next_to(nums[-3], m.RIGHT, buff=0.24)
        y2=math_tex("1").set_color(YELLOW).scale(0.19).next_to(nums[-2], m.RIGHT, buff=0.24)
        y3=math_tex("0").set_color(GREEN).scale(0.19).next_to(nums[-1], m.RIGHT, buff=0.24)

        self.play(m.FadeIn(y1))
        self.play(m.FadeIn(y2))
        self.play(m.FadeIn(y3))
        self.wait()

        self.play(m.Create(p34_blocks[2]))
        self.remove(p34_blocks[2])
        self.wait()


        minus_1=math_tex("-").scale(0.2).next_to(nums[-3], m.RIGHT, buff=0.09)
        equals_1=math_tex("=").scale(0.2).next_to(y1, m.RIGHT, buff=0.12)
        d_1=math_tex("0.91").set_color("#00FFFF").scale(0.18).next_to(y1, m.RIGHT, buff=0.28)

        minus_2=math_tex("-").scale(0.2).next_to(nums[-2], m.RIGHT, buff=0.09)
        equals_2=math_tex("=").scale(0.2).next_to(y2, m.RIGHT, buff=0.12)
        d_2=math_tex("-0.91").set_color(YELLOW).scale(0.18).next_to(y2, m.RIGHT, buff=0.28)

        minus_3=math_tex("-").scale(0.2).next_to(nums[-1], m.RIGHT, buff=0.09)
        equals_3=math_tex("=").scale(0.2).next_to(y3, m.RIGHT, buff=0.12)
        d_3=math_tex("0.00").set_color(GREEN).scale(0.18).next_to(y3, m.RIGHT, buff=0.28)
        error_equations=m.VGroup(minus_1, equals_1, d_1, minus_2, equals_2, d_2, minus_3, equals_3, d_3)


        self.play(m.FadeIn(p34_blocks[1][-1]), m.FadeOut(dldh_border),
                  dldh_eq.animate.set_color(CHILL_BROWN).scale(0.75).move_to([0.93, 0.58, 0]))
        self.play(m.Write(minus_1), m.Write(equals_1), m.Write(d_1))
        self.wait()

        d_1_copy=d_1.copy()
        d_1_copy.scale(0.85)
        d_1_copy.move_to([1.61, 0.57, 0])

        self.play(m.Create(p34_blocks[4]), self.animate_view(0, 0, 0, (0.03, -0.01, 0.0), 2.05))
        self.play(m.ReplacementTransform(d_1.copy(), d_1_copy))
        self.wait()

        d_2_copy=d_2.copy()
        d_2_copy.scale(0.85)
        d_2_copy.move_to([1.66, 0.08, 0])

        self.play(m.Write(minus_2), m.Write(equals_2), m.Write(d_2))
        self.wait()
        self.play(m.Create(p34_blocks[5]))
        self.play(m.ReplacementTransform(d_2.copy(), d_2_copy))
        self.wait()

        self.play(m.FadeIn(minus_3), m.FadeIn(equals_3), m.FadeIn(d_3))
        self.wait()

        #p35
        self.play(rect_1.animate.set_opacity(0.3), rect_2.animate.set_opacity(0.3), 
                  m.FadeOut(p34_blocks[5]), m.FadeOut(p34_blocks[4]), m.FadeOut(d_1_copy), m.FadeOut(d_2_copy),
                  self.animate_view(0, 0, 0, (-0.17, -0.07, 0.0), 1.81))



        dldmb=math_tex(r"\frac{\partial L}{\partial m_2}")
        modular_eq_1b=math_tex("=")
        modular_eq_2b=math_tex(r"\frac{\partial h_2}{\partial m_2}")
        modular_eq_3b=math_tex(r"\cdot")
        modular_eq_4b=math_tex(r"\frac{\partial L}{\partial h_2}")

        modular_eqb=m.VGroup(dldmb, modular_eq_1b, modular_eq_2b, modular_eq_3b, modular_eq_4b)
        dldmb.scale(0.17)
        modular_eq_1b.scale(0.22)
        modular_eq_2b.scale(0.17)
        modular_eq_3b.scale(0.35)
        modular_eq_4b.scale(0.17)
        dldmb.move_to([-1.5, -0.7, 0])
        modular_eq_1b.next_to(dldmb, m.RIGHT, buff=0.15)
        modular_eq_2b.next_to(modular_eq_1b, m.RIGHT, buff=0.14)
        modular_eq_3b.next_to(modular_eq_2b, m.RIGHT, buff=0.35)
        modular_eq_4b.next_to(modular_eq_3b, m.RIGHT, buff=0.33)

        self.play(m.FadeIn(modular_eqb))
        self.wait()

        cross_out_line=m.Line(m.ORIGIN, m.RIGHT*0.3).set_stroke(color=YELLOW, width=3)
        cross_out_line.rotate(m.DEGREES*45)
        cross_out_line.move_to(modular_eq_4b)

        y_minus_yhat=math_tex("(\hat{y}_2-y_2)").scale(0.18).set_color(YELLOW)
        y_minus_yhat.next_to(modular_eq_4b, m.RIGHT, buff=0.05)


        self.play(m.ReplacementTransform(dldh_eq[-4:].copy(), y_minus_yhat[1:-1]), run_time=2.0)
        self.add(y_minus_yhat)
        self.play(m.Create(cross_out_line))
        self.wait()

        self.play(rect_1.animate.set_opacity(0.0),rect_2.animate.set_opacity(0.0), 
                  m.FadeOut(nums[1]), m.FadeOut(nums[3]), m.FadeOut(nums[4]), m.FadeOut(nums[6]),
                  m.FadeOut(nums[7]), m.FadeOut(nums[9:]), m.FadeOut(y1), m.FadeOut(y2), m.FadeOut(y3), m.FadeOut(p34_blocks[0]),
                  m.FadeOut(error_equations), m.FadeOut(p34_blocks[1]), m.FadeOut(dldh_eq), m.FadeOut(p35_net_background[0]),
                  self.animate_view(0, 0, 0, (-0.57, -0.14, 0.0), 1.76))
        self.wait()

        h2_eq=math_tex("h_2=m_2x+b_2").scale(0.2)
        h2_eq.move_to([-0.9, -0.3, 0])

        self.play(m.ReplacementTransform(h2_label.copy(), h2_eq[:2]))
        self.add(h2_eq[2])
        self.play(m.ReplacementTransform(m2_label.copy(), h2_eq[3:5]), m.ReplacementTransform(x_label.copy(), h2_eq[5]))
        self.add(h2_eq[6])
        self.play(m.ReplacementTransform(b2_label.copy(), h2_eq[7:]))
        self.wait()

        p36_blocks=m.VGroup()
        for p in sorted(glob.glob(svg_path+'/p36/*.svg')):
            p36_blocks.add(m.SVGMobject(p)[1:])
        p36_blocks[0].move_to([-0.9, 0.13, 0])    

        x_labelb=math_tex('x').set_color(CHILL_BROWN).scale(0.2)
        x_labelb.move_to([-0.55, -0.2, 0])
        m2_labelb=math_tex('m_2').set_color(YELLOW).scale(0.2)
        m2_labelb.move_to([-0.77, 0.19, 0])
        b2_labelb=math_tex('b_2').set_color(YELLOW).scale(0.2)
        b2_labelb.move_to([-1.26, -0.05, 0])
        h2_labelb=math_tex('h_2').set_color(CHILL_BROWN).scale(0.2)
        h2_labelb.move_to([-1.27, 0.48, 0])
        manual_labels_b=m.VGroup(x_labelb, m2_labelb, b2_labelb, h2_labelb)

        self.wait(0)
        self.remove(nums[0], nums[2], nums[5], nums[8], p35_net_background[1], manual_background_labels)
        self.play(m.Create(p36_blocks[0]))
        self.play(m.ReplacementTransform(h2_eq[:2].copy(), h2_labelb),m.ReplacementTransform(h2_eq[5].copy(), x_labelb))
        self.play(m.ReplacementTransform(h2_eq[3:5].copy(), m2_labelb))
        self.play(m.ReplacementTransform(h2_eq[-2:].copy(), b2_labelb))
        self.wait()

        box = m.SurroundingRectangle(modular_eq_2b, color=YELLOW, buff=0.025)
        self.play(m.Create(box))
        self.wait()
        self.play(m.FadeOut(box))

        plot_2=p36_blocks[0].copy()
        plot_2.move_to([0.0, 0.13, 0])

        x_labelc=math_tex('x').set_color(YELLOW).scale(0.2)
        x_labelc.move_to([-0.77+0.87, 0.19, 0])
        m2_labelc=math_tex('m_2').set_color(CHILL_BROWN).scale(0.2)
        m2_labelc.move_to([-0.55+0.93, -0.2, 0])
        b2_labelc=math_tex('b_2').set_color(YELLOW).scale(0.2)
        b2_labelc.move_to([-1.26+0.9, -0.05, 0])
        h2_labelc=math_tex('h_2').set_color(CHILL_BROWN).scale(0.2)
        h2_labelc.move_to([-1.27+0.9, 0.48, 0])
        manual_labels_c=m.VGroup(x_labelc, m2_labelc, b2_labelc, h2_labelc)

        h2_eq_copy=h2_eq.copy().move_to([0, -0.3, 0])


        self.play(m.ReplacementTransform(p36_blocks[0].copy(), plot_2),
                 m.ReplacementTransform(x_labelb.copy(), x_labelc),
                 m.ReplacementTransform(m2_labelb.copy(), m2_labelc),
                 m.ReplacementTransform(b2_labelb.copy(), b2_labelc),
                 m.ReplacementTransform(h2_labelb.copy(), h2_labelc),
                 m.ReplacementTransform(h2_eq.copy(), h2_eq_copy), run_time=2.5)


        #Ok now I need ot cross out dh/dmw, and bring x ver from the second plot!
        self.wait()

        x_eq=x_labelc.copy().next_to(modular_eq_2b, m.RIGHT, buff=0.05)

        cross_out_line_2=m.Line(m.ORIGIN, m.RIGHT*0.3).set_stroke(color=YELLOW, width=3)
        cross_out_line_2.rotate(m.DEGREES*45)
        cross_out_line_2.move_to(modular_eq_2b)

        self.play(m.ReplacementTransform(x_labelc, x_eq), run_time=2)
        self.play(m.Create(cross_out_line_2))
        self.wait()

        #Start p37
        self.remove(net_background)
        net_background.set_opacity(1.0)
        # self.add(net_background)
        self.play(m.FadeOut(manual_labels_c), m.FadeOut(manual_labels_b), m.FadeOut(plot_2), m.FadeOut(p36_blocks[0]),
                  m.FadeOut(h2_eq), m.FadeOut(h2_eq_copy))
        self.play(rect_1.animate.set_opacity(0.2),rect_2.animate.set_opacity(0.2), m.FadeIn(nums),
                  m.FadeIn(net_background), m.FadeIn(y1), m.FadeIn(y2), m.FadeIn(y3), m.FadeIn(p34_blocks[0]),
                  m.FadeIn(error_equations), m.FadeIn(p34_blocks[1]), m.FadeIn(dldh_eq),
                  self.animate_view(0, 0, 0, (-0.26, -0.07, 0.0), 1.75))
        self.wait()

        box3 = m.SurroundingRectangle(nums[2], color=YELLOW, buff=0.025)
        self.play(m.Create(box3))
        self.wait()
        box4 = m.SurroundingRectangle(nums[0], color=YELLOW, buff=0.025)
        self.play(m.Create(box4))
        self.wait()
        self.play(m.FadeOut(box3), m.FadeOut(box4))
        self.wait()

        #So i was foing to do a tick the number up/tick the number down thing - but I'm not sure it actually helps make the p37 point. Let's keep moving. 
        #Start p38

        dldm2_final=math_tex(r"\frac{\partial L}{\partial m_2} \hspace{1mm} = \hspace{1mm} x \hspace{1mm} (\hat{y}_2-y)").scale(0.17)
        dldm2_final[7:].set_color(YELLOW)
        dldm2_final.next_to(dldmb, m.RIGHT, buff=0.1)

        self.play(m.FadeOut(cross_out_line), m.FadeOut(cross_out_line_2), m.FadeOut(modular_eq_2b), m.FadeOut(modular_eq_3b) ,m.FadeOut(modular_eq_4b))
        self.play(m.ReplacementTransform(dldmb, dldm2_final[:6]),
                  # ReplacementTransform(modular_eq_1b, dldm2_final[6]),
                  modular_eq_1b.animate.move_to(dldm2_final[6]),
                  # ReplacementTransform(x_eq, dldm2_final[7]),
                  # ReplacementTransform(y_minus_yhat, dldm2_final[8:])
                  x_eq.animate.move_to(dldm2_final[7]),
                  y_minus_yhat.animate.next_to(dldm2_final[7], m.RIGHT, buff=0.05).shift(0.007*m.UP), run_time=2.5)

        self.wait()

        equals_4=math_tex("=").scale(0.17)
        equals_4.next_to(y_minus_yhat, buff=0.05)
        
        x_copy=nums[0].copy()
        self.play(x_copy.animate.next_to(equals_4, buff=0.07).scale(1.35).shift(0.007*m.UP), run_time=1.2)
        self.add(equals_4)
        self.wait()

        paren_1=math_tex("(").scale(0.17).set_color(YELLOW)
        paren_1.next_to(x_copy, buff=0.05)
        yhat_copy=nums[-2].copy()
        self.play(yhat_copy.animate.next_to(paren_1, buff=0.03).scale(1.1), run_time=1.2)
        self.add(paren_1)

        minus_4=math_tex("-").scale(0.17).set_color(YELLOW)
        minus_4.next_to(yhat_copy, buff=0.05)
        y2_copy=y2.copy()
        self.play(y2_copy.animate.next_to(minus_4, buff=0.03).scale(1.00), run_time=1.2)
        self.add(minus_4)

        paren_2=math_tex(")").scale(0.17).set_color(YELLOW)
        paren_2.next_to(y2_copy, buff=0.03)
        self.add(paren_2)

        self.wait()
        result=math_tex("=-2.140").scale(0.185)
        result.next_to(paren_2, buff=0.05)
        self.play(m.Write(result))
        self.wait()

        little_m2_loss_plot=m.SVGMobject(svg_path+'/little_m2_loss_plot_1.svg')
        little_m2_loss_plot.scale(0.2)
        little_m2_loss_plot.next_to(result, m.RIGHT, buff=0.04)

        self.play(m.Create(little_m2_loss_plot))
        result_copy=result[1:].copy()
        self.play(result_copy.animate.move_to([1.0, -0.65, 0]).scale(0.75).set_color(YELLOW))
        self.wait()

        #ok so i think we land on the  6 partial derivatives I do a clean start/break at p40?
        big_final_eq=m.VGroup(dldm2_final[:6], equals_4, x_eq, x_copy, paren_1, minus_4, paren_2, result, y2_copy, yhat_copy, y_minus_yhat, result_copy)
        full_gradient=math_tex(r"\left[ \frac{\partial L}{\partial m_1}, \hspace{1mm} \frac{\partial L}{\partial m_2}, \hspace{1mm} \frac{\partial L}{\partial m_3}, \hspace{1mm} \frac{\partial L}{\partial b_1}, \hspace{1mm} \frac{\partial L}{\partial b_2}, \hspace{1mm} \frac{\partial L}{\partial b_3} \right]")
        full_gradient.scale(0.2)

        self.play(m.ReplacementTransform(dldm2_final[:6].copy(), full_gradient[8:14]),
                  rect_1.animate.set_opacity(0.0),
                  rect_2.animate.set_opacity(0.0), 
                  m.FadeOut(nums),
                  m.FadeOut(net_background),
                  m.FadeOut(y1),
                  m.FadeOut(y2),
                  m.FadeOut(y3),
                  m.FadeOut(p34_blocks[0]),
                  m.FadeOut(error_equations),
                  m.FadeOut(p34_blocks[1]),
                  m.FadeOut(dldh_eq),
                  m.FadeOut(big_final_eq),
                  m.FadeOut(little_m2_loss_plot),
                  m.FadeOut(modular_eq_1b),
                  m.FadeOut(result_copy),
                  self.animate_view(0, 0, 0, (0.0, 0.0, 0.0), 1.75),
                  run_time=3)
        self.add(full_gradient)
        self.wait()

        gradient_label=m.SVGMobject(svg_path+'/gradient_label.svg')
        gradient_label.scale(0.083)
        gradient_label.move_to([0,-0.27,0])

        self.play(m.FadeIn(gradient_label))
        # self.play(ShowCreation(gradient_label))
        self.wait()
        self.play(m.FadeOut(gradient_label))


        self.wait(20)
        pass















