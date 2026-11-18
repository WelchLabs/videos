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


class p40_44_v2(WelchScene):
    def construct(self):
        '''
        
        '''
        # net_background=SVGMobject(svg_path+'/p44_background_1.svg')[1:]
        net_background=m.SVGMobject(svg_path+'/p40_net_background_manim.svg')[1:]

        
        x=2.3514
        w=[1, 0, -1, 0, 0, 0]
        logits=[2.35, 0, -2.35]
        yhats=[0.91, 0.09, 0.00]
        nums=get_numbers_3(x, w, logits, yhats)
    
        m1_label=math_tex('m_1').set_color(CHILL_BROWN).scale(0.11)
        m1_label.move_to([-1.19, 0.14, 0])
        b1_label=math_tex('b_1').set_color(CHILL_BROWN).scale(0.11)
        b1_label.move_to([-0.87, 0.29, 0])
        m2_label=math_tex('m_2').set_color(CHILL_BROWN).scale(0.11)
        m2_label.move_to([-1.155, -0.05, 0])
        b2_label=math_tex('b_2').set_color(CHILL_BROWN).scale(0.11)
        b2_label.move_to([-0.87, -0.055, 0])
        m3_label=math_tex('m_3').set_color(CHILL_BROWN).scale(0.11)
        m3_label.move_to([-1.16, -0.23, 0])
        b3_label=math_tex('b_3').set_color(CHILL_BROWN).scale(0.11)
        b3_label.move_to([-0.87, -0.4, 0])
        manual_labels=m.VGroup(m1_label, b1_label, m2_label, b2_label, m3_label, b3_label)
        # self.add(net_background, nums, manual_labels)

        full_gradient=math_tex(r"\left[ \frac{\partial L}{\partial m_1}, \hspace{1mm} \frac{\partial L}{\partial m_2}, \hspace{1mm} \frac{\partial L}{\partial m_3}, \hspace{1mm} \frac{\partial L}{\partial b_1}, \hspace{1mm} \frac{\partial L}{\partial b_2}, \hspace{1mm} \frac{\partial L}{\partial b_3} \right]")
        full_gradient.scale(0.2)
        self.add(full_gradient)
        self.set_view(0, 0, 0, (0.0, 0.0, 0.0), 1.75)

        grade_eq_scale=0.15
        grad_1=math_tex(r"\frac{\partial L}{\partial m_1} = x(\hat{y}_1-y_1)=2.14").scale(grade_eq_scale)
        grad_2=math_tex(r"\frac{\partial L}{\partial m_2} = x(\hat{y}_2-y_2)=-2.14 ").scale(grade_eq_scale)
        grad_3=math_tex(r"\frac{\partial L}{\partial m_3} = x(\hat{y}_3-y_3)=0.00 ").scale(grade_eq_scale)
        grad_4=math_tex(r"\frac{\partial L}{\partial b_1} = (\hat{y}_1-y_1)=0.91").scale(grade_eq_scale)
        grad_5=math_tex(r"\frac{\partial L}{\partial b_2} = (\hat{y}_2-y_2)=-0.91").scale(grade_eq_scale)
        grad_6=math_tex(r"\frac{\partial L}{\partial b_3} = (\hat{y}_3-y_3)=0.00" ).scale(grade_eq_scale)
        grad_eqs=m.VGroup(grad_1, grad_2, grad_3, grad_4, grad_5, grad_6)
        grad_1[-4:].set_color('#00FFFF')
        grad_2[-5:].set_color(YELLOW)
        grad_3[-4:].set_color(GREEN)
        grad_4[-4:].set_color('#00FFFF')
        grad_5[-5:].set_color(YELLOW)
        grad_6[-4:].set_color(GREEN)

        vertical_spacing=0.2
        x_start=0.55
        for i, g in enumerate(grad_eqs):
            g.move_to([x_start, vertical_spacing*(2.5-i), 0], aligned_edge=m.LEFT)

        self.wait()

        self.play(m.ReplacementTransform(full_gradient[1:7], grad_1[:6]),
                  m.ReplacementTransform(full_gradient[8:14], grad_2[:6]),
                  m.ReplacementTransform(full_gradient[15:21], grad_3[:6]),
                  m.ReplacementTransform(full_gradient[22:28], grad_4[:6]),
                  m.ReplacementTransform(full_gradient[29:35], grad_5[:6]),
                  m.ReplacementTransform(full_gradient[36:42], grad_6[:6]),
                  m.FadeOut(full_gradient[0]),
                  m.FadeOut(full_gradient[7]),
                  m.FadeOut(full_gradient[14]),
                  m.FadeOut(full_gradient[21]),
                  m.FadeOut(full_gradient[28]),
                  m.FadeOut(full_gradient[35]),
                  m.FadeOut(full_gradient[42]),
                  m.FadeIn(net_background),
                  m.FadeIn(manual_labels),
                  m.FadeIn(nums),
                  self.animate_view(0, 0, 0, (-0.1, -0.02, 0.0), 1.94),
                  run_time=2.4
                  )

        self.wait()
        self.add(grad_2[6:-6])
        self.wait()
        self.play(m.Write(grad_1[6:-5]),
                  m.Write(grad_3[6:-5]),
                  m.Write(grad_4[6:-5]),
                  m.Write(grad_5[6:-6]),
                  m.Write(grad_6[6:-5]),
             )
        self.wait()


        
        vector_eqs=m.SVGMobject(svg_path+'/vector_eqs_1.svg')
        vector_eqs.scale(0.58)
        vector_eqs.move_to([1.4,0,0])

        self.play(m.Create(vector_eqs),
                 self.animate_view(0, 0, 0, (0.01, -0.04, 0.0), 1.96))
        self.wait()

        self.play(m.FadeOut(vector_eqs))
        self.play(m.Write(grad_1[-5:]),
                m.Write(grad_2[-6:]),
                m.Write(grad_3[-5:]),
                m.Write(grad_4[-5:]),
                m.Write(grad_5[-6:]),
                m.Write(grad_6[-5:]))
        

        p42_blocks=m.VGroup()
        for p in sorted(glob.glob(svg_path+'/p42/*.svg')):
            p42_blocks.add(m.SVGMobject(p)[1:])
        p42_blocks.move_to([-0.01, 0.13, 0])
        self.wait()

        grad_2_val=grad_2[-5:].copy()
        self.play(m.FadeIn(p42_blocks[1]), self.animate_view(0, 0, 0, (-0.12, 0.23, 0.0), 1.94), run_time=2.0)
        self.play(m.Create(p42_blocks[3]), grad_2_val.animate.move_to([0.15, 0.85, 0]), run_time=1.5)
        self.wait()

        box=m.SurroundingRectangle(nums[-2], color=YELLOW, buff=0.025)
        self.play(m.Create(box))
        self.wait()

        w=[1, 0.1, -1, 0, 0, 0]
        logits=[2.35, 0.24, -2.35]
        yhats=[0.89, 0.11, 0.00]
        nums2=get_numbers_3(x, w, logits, yhats)

        box2=m.SurroundingRectangle(nums[2], color=YELLOW, buff=0.025)
        self.play(m.Create(box2))
        self.wait()

        self.play(m.ReplacementTransform(nums[2], nums2[2]))
        self.play(m.ReplacementTransform(nums[8], nums2[8]),
                  m.ReplacementTransform(nums[11], nums2[11]),
                  m.ReplacementTransform(nums[10], nums2[10]),
                  m.ReplacementTransform(nums[12], nums2[12]))
        self.wait()
        self.play(m.FadeOut(box), m.FadeOut(box2))
        self.wait()

        box3=m.SurroundingRectangle(grad_1[-4:], color="#00FFFF", buff=0.025)
        self.play(m.Create(box3))
        self.wait()

        grad_1_val=grad_1[-4:].copy()
        self.play(m.FadeIn(p42_blocks[0]))
        self.play(m.Create(p42_blocks[2]), grad_1_val.animate.move_to([-0.45, 0.855, 0]), run_time=1.5)
        self.wait()

        w=[1.1, 0.1, -1, 0, 0, 0]
        logits=[2.59, 0.24, -2.35]
        yhats=[0.91, 0.09, 0.00]
        nums3=get_numbers_3(x, w, logits, yhats)

        box4=m.SurroundingRectangle(nums[1], color="#00FFFF", buff=0.025)
        self.play(m.Create(box4))

        self.remove(nums[1], nums[7], nums[10:12])
        self.add(nums2[1], nums2[7], nums2[10:12])
        self.wait()
        self.play(m.ReplacementTransform(nums2[1], nums3[1]))
        self.play(m.ReplacementTransform(nums2[7], nums3[7]),
                  m.ReplacementTransform(nums2[11], nums3[11]),
                  m.ReplacementTransform(nums2[10], nums3[10]),
                  m.ReplacementTransform(nums2[12], nums3[12]))
        self.wait()
        self.play(m.FadeOut(box3), m.FadeOut(box4))
        self.wait()

        self.play(m.Create(p42_blocks[4]))
        self.wait()

        self.play(m.Create(p42_blocks[5]))
        self.wait()       
        # Hmm yeah showing all the numbers ticking around as I change m2 would be pretty cool
        # Same thing I do when visualizing training right?
        # Hmm actually I think the numbers ticking right now is going to be ditracting - > but I do 
        # think changing the number once will make sense/be helpful - easier to animate too - let's try that. 

        #Ok time to gear up to move to big gradient descent equation
        #Jump to end and then reconstruct here. Actually let me sort out my fade outs real quick first. 
        # Ok so i probably want to make vectors including brackets in manim, and bring in other stuff from illustraotr
        p43_blocks=m.VGroup()
        for p in sorted(glob.glob(svg_path+'/p_43/*.svg')):
            p43_blocks.add(m.SVGMobject(p)[1:])
        p43_blocks.move_to([-0.3,-0.25,0])
        p43_blocks.scale(0.7)

        params_vec = math_tex(r"\begin{bmatrix} m_1 \\ m_2 \\ m_3 \\ b_1 \\ b_2 \\ b_3 \end{bmatrix}")
        grads_vec = math_tex(r"\begin{bmatrix} \partial L / \partial m_1 \\ \partial L / \partial m_2 \\ \partial L / \partial m_3 \\ \partial L / \partial b_1 \\ \partial L / \partial b_2 \\ \partial L / \partial b_3 \\ \end{bmatrix}")

        
        params_vec.scale(0.17)
        params_vec.move_to([-0.88, 0, 0])
        params_vec_2=params_vec.copy()
        params_vec_2.move_to([0.32, 0, 0])

        grads_vec.scale(0.17)
        grads_vec.move_to([-0.24, 0, 0])

        self.play(m.FadeOut(net_background),
                  m.FadeOut(nums[0]),
                  m.FadeOut(nums3[1]),
                  m.FadeOut(nums2[2]),
                  m.FadeOut(nums[3]),
                  m.FadeOut(nums[4:7]),
                  m.FadeOut(nums3[7]),
                  m.FadeOut(nums2[8]),
                  m.FadeOut(nums[9]),
                  m.FadeOut(nums3[10]),
                  m.FadeOut(nums3[11]),
                  m.FadeOut(nums3[12]),
                  m.FadeOut(p42_blocks),
                  m.FadeOut(grad_1_val),
                  m.FadeOut(grad_2_val),
            )
        self.play(m.ReplacementTransform(m1_label, params_vec[8:10]),
                  m.ReplacementTransform(m2_label, params_vec[10:12]),
                  m.ReplacementTransform(m3_label, params_vec[12:14]),
                  m.ReplacementTransform(b1_label, params_vec[14:16]),
                  m.ReplacementTransform(b2_label, params_vec[16:18]),
                  m.ReplacementTransform(b3_label, params_vec[18:20]),
                  self.animate_view(0, 0, 0, (0.11, -0.01, 0.0), 1.61),
                  run_time=2
            )
        self.add(params_vec)
        self.add(p43_blocks[0])
        self.wait()

        self.play(m.FadeOut(grad_1[6:]),
              m.FadeOut(grad_2[6:]),
              m.FadeOut(grad_3[6:]),
              m.FadeOut(grad_4[6:]),
              m.FadeOut(grad_5[6:]),
              m.FadeOut(grad_6[6:]),
              m.ReplacementTransform(grad_1[:6], grads_vec[8:14]),
              m.ReplacementTransform(grad_2[:6], grads_vec[14:20]),
              m.ReplacementTransform(grad_3[:6], grads_vec[20:26]),
              m.ReplacementTransform(grad_4[:6], grads_vec[26:32]),
              m.ReplacementTransform(grad_5[:6], grads_vec[32:38]),
              m.ReplacementTransform(grad_6[:6], grads_vec[38:44]),
              self.animate_view(0, 0, 0, (-0.24, -0.08, 0.0), 1.61),
              run_time=2
              ) 

        self.play(m.FadeIn(grads_vec[:8]),
                 m.FadeIn(grads_vec[44:]),
                 m.FadeIn(p43_blocks[1]),
                 m.FadeIn(p43_blocks[3]),
                 m.FadeIn(p43_blocks[4]),
                 m.FadeIn(params_vec_2)
            )  
        self.wait()
        self.play(m.FadeIn(p43_blocks[2]))




        self.wait()
        








        self.wait(20)
        pass

