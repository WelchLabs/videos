import manimgx as m
import numpy as np
from welch_scene import WelchScene
from welch_text import math_tex
from functools import partial

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
BLUE='#65c8d0'


confidences=[1e-07, 2.9e-06, 0.000107, 0.0028635, 0.0505046, 0.3706285, 0.738614, 0.8713107, 0.9219979, 0.9469875, 0.9613259, 0.9703088, 0.9762792, 0.9804361, 0.9834525, 0.9857258, 0.9875003, 0.9889282, 0.9901052, 0.991092, 0.9919273, 0.9926376, 0.9932413, 0.9937531, 0.9941856, 0.9945493, 0.9948549, 0.9951125, 0.9953318, 0.9955214, 0.9956887, 0.9958402, 0.9959794, 0.9961088, 0.9962296, 0.9963422, 0.9964456, 0.9965395, 0.9966245, 0.9966999, 0.9967668, 0.9968262, 0.996879, 0.9969254, 0.9969669, 0.9970031, 0.9970347, 0.9970604, 0.9970797, 0.9970914, 0.9970949, 0.9970886, 0.9970725, 0.9970466, 0.9970107, 0.9969663, 0.9969141, 0.9968557, 0.9967931, 0.9967289, 0.9966677, 0.9966158, 0.9965843, 0.996585, 0.9966252, 0.9967, 0.996796, 0.9968995, 0.9970031, 0.9971027, 0.9971969, 0.9972847, 0.9973665, 0.9974422, 0.9975125, 0.9975783, 0.9976388, 0.997696, 0.9977484, 0.9977974, 0.9978439, 0.9978867, 0.9979271, 0.9979651, 0.9980007, 0.9980342, 0.998066, 0.9980965, 0.9981249, 0.998152, 0.9981775, 0.998202, 0.9982254, 0.9982477, 0.9982692, 0.9982898, 0.9983097, 0.9983286, 0.9983473, 0.9983647, 0.998382, 0.9983988, 0.998415, 0.9984306, 0.998446, 0.9984611, 0.9984761, 0.9984907, 0.9985049, 0.9985185, 0.9985322, 0.9985459, 0.9985593, 0.9985722, 0.9985851, 0.9985977, 0.9986104, 0.9986233, 0.9986355, 0.9986481, 0.9986601, 0.9986722, 0.998684, 0.998696, 0.9987079, 0.9987195, 0.9987312, 0.9987424, 0.9987538, 0.9987651, 0.9987761, 0.9987873, 0.9987983, 0.9988091, 0.9988198, 0.9988305, 0.9988411, 0.9988515, 0.9988617, 0.998872, 0.9988819, 0.9988919, 0.9989021, 0.9989117, 0.9989214, 0.998931, 0.9989403, 0.9989496, 0.9989587, 0.9989678, 0.9989766, 0.9989854, 0.9989941, 0.9990024, 0.9990108, 0.9990189, 0.9990273, 0.9990356, 0.9990432, 0.9990512, 0.9990589, 0.9990664, 0.9990737, 0.9990811, 0.9990882, 0.9990953, 0.9991025, 0.9991093, 0.999116, 0.9991228, 0.9991296, 0.9991361, 0.9991424, 0.9991488, 0.999155, 0.9991611, 0.9991671, 0.999173, 0.9991789, 0.9991845, 0.9991906, 0.9991964, 0.999202, 0.9992074, 0.9992127, 0.9992181, 0.9992234, 0.9992287, 0.9992338, 0.9992387, 0.9992441, 0.999249, 0.9992538, 0.9992587, 0.9992636, 0.9992684, 0.9992729, 0.9992774, 0.9992822, 0.9992867, 0.9992909, 0.9992954, 0.9992999, 0.9993041, 0.9993085, 0.9993128, 0.9993169, 0.9993212, 0.9993254, 0.9993293, 0.9993333, 0.9993374, 0.9993414, 0.9993451, 0.9993489, 0.9993528, 0.9993564, 0.9993604, 0.999364, 0.9993677, 0.9993713, 0.9993747, 0.9993783, 0.9993819, 0.9993854, 0.9993889, 0.9993922, 0.9993956, 0.9993991, 0.9994023, 0.9994057, 0.9994091, 0.9994121, 0.9994153, 0.9994184, 0.9994216, 0.9994249, 0.999428, 0.999431, 0.9994339, 0.9994369, 0.9994399, 0.999443, 0.9994459, 0.9994488, 0.9994517, 0.9994547, 0.9994575, 0.9994603, 0.9994631, 0.9994658, 0.9994686, 0.9994715, 0.9994743, 0.9994769, 0.9994795]

class p48b(WelchScene):
    def construct(self):
        '''
        Render out nice manim numbers to 4 decimal places for the confidence and cross entropy loss, where 
        loss = -ln(confidence)
        '''
        
        # Create labels with MathTex for LaTeX style
        conf_label = math_tex(r"\text{Confidence:}", color=m.WHITE, font_size=100)
        loss_label = math_tex(r"\text{Loss:}", color=m.WHITE, font_size=100)
        
        # Position labels on the left side of the screen
        conf_label.shift(m.UP * 1 + m.LEFT * 3)
        loss_label.shift(m.DOWN * 1 + m.LEFT * 3)
        
        # Initial confidence and loss values with Tex
        conf_value = math_tex(f"{confidences[0]:.4f}", color=m.WHITE, font_size=100)
        loss_value = math_tex(f"{-np.log(confidences[0]):.4f}", color=m.WHITE, font_size=100)
        
        # Position values to the right of their labels
        conf_value.next_to(conf_label, m.RIGHT, buff=0.5)
        loss_value.next_to(loss_label, m.RIGHT, buff=0.5)
        
        # Add everything to the scene
        self.add(conf_label, loss_label, conf_value, loss_value)
        self.wait(1)
        
        # Update the values with instant replacement (no animation transitions)
        for i in range(1, len(confidences)):
            # Remove previous values
            self.remove(conf_value, loss_value)
            
            # Create new values
            conf_value = math_tex(f"{confidences[i]:.4f}", color=m.WHITE, font_size=100)
            loss_value = math_tex(f"{-np.log(confidences[i]):.4f}", color=m.WHITE, font_size=100)
            
            # Position them
            conf_value.next_to(conf_label, m.RIGHT, buff=0.5)
            loss_value.next_to(loss_label, m.RIGHT, buff=0.5)
            
            # Add new values
            self.add(conf_value, loss_value)
            self.wait(1)
        
        self.wait(2)




# confidences=[0.3916, 0.3412, 0.8123, 0.9531] #Fake data, replace later with real

# class p48b(InteractiveScene):
#     def construct(self):
#         '''
#         Render out nice manim numbers to 4 decimal places for the confidence and cross entropy loss, where 
#         loss = -ln(confidence)
#         '''
#         # Create labels
#         conf_label = Text("Confidence:", color=YELLOW, font_size=54)
#         loss_label = Text("Loss:", color=BLUE, font_size=36)
        
#         # Position labels on the left side of the screen
#         conf_label.shift(UP * 1 + LEFT * 3)
#         loss_label.shift(DOWN * 1 + LEFT * 3)
        
#         # Create decimal numbers with 4 decimal places
#         conf_value = DecimalNumber(
#             confidences[0],
#             num_decimal_places=4,
#             include_sign=False,
#             font_size=36,
#             color=YELLOW
#         )
        
#         loss_value = DecimalNumber(
#             -np.log(confidences[0]),
#             num_decimal_places=4,
#             include_sign=False,
#             font_size=36,
#             color=BLUE
#         )
        
#         # Position values to the right of their labels
#         conf_value.next_to(conf_label, RIGHT, buff=0.5)
#         loss_value.next_to(loss_label, RIGHT, buff=0.5)
        
#         # Add everything to the scene
#         self.play(
#             Write(conf_label),
#             Write(loss_label),
#             FadeIn(conf_value),
#             FadeIn(loss_value)
#         )
#         self.wait(1)
        
#         # Update the values sequentially for each confidence in our list
#         for i in range(1, len(confidences)):
#             self.play(
#                 conf_value.animate.set_value(confidences[i]),
#                 loss_value.animate.set_value(-np.log(confidences[i])),
#                 run_time=1.5
#             )
#             self.wait(1)
        
#         # Example of cycling through the values repeatedly
#         for _ in range(2):  # Cycle through twice
#             for i in range(len(confidences)):
#                 self.play(
#                     conf_value.animate.set_value(confidences[i]),
#                     loss_value.animate.set_value(-np.log(confidences[i])),
#                     run_time=1
#                 )
#                 self.wait(0.5)
        
#         self.wait(2)



