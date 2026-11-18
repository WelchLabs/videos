import manimgx as gx
from welch_scene import WelchScene
from tqdm import tqdm
from pathlib import Path

CHILL_BROWN='#948979'
YELLOW='#ffd35a'
YELLOW_FADE='#7f6a2d'
BLUE='#65c8d0'
GREEN='#00a14b' #6e9671' 
CHILL_GREEN='#6c946f'
CHILL_BLUE='#3d5c6f'
FRESH_TAN='#dfd0b9'
CYAN='#00FFFF'
MAGENTA='#FF00FF'

class IntegrationProblems(WelchScene):
    def construct(self):



        eq1=gx.MathTex(r'\int_0^{2\pi} \sin(x) \cos(x) \tan(x) \cot(x) \sec(x) \csc(x) \, dx')
        eq2=gx.MathTex(r'\int \sqrt[3]{x \sqrt[4]{x \sqrt[5]{x \sqrt[6]{x \sqrt{\cdots}}}}} \, dx')
        eq3=gx.MathTex(r'\int_0^1 \left( \sum_{k=1}^{\infty} (-1)^k x^{2k} \right) dx')

        # self.add(eq1)
        self.play(gx.Write(eq1), run_time=4)
        self.wait()
        self.play(gx.FadeOut(eq1))

        self.play(gx.Write(eq2), run_time=4)
        self.wait()
        self.play(gx.FadeOut(eq2))

        self.play(gx.Write(eq3), run_time=4)
        self.wait()
        self.play(gx.FadeOut(eq3))

        
        self.wait(20)
