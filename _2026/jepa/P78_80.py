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

#ulimit -n 4096

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/graphics/p78_80')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa/hacking/v_jepa_ac_1'

def make_emb_tex(vals, color, position, scale=0.65):
    s = (r'\begin{bmatrix} '
         + f'{vals[0]:.2f}' + r', \ '
         + f'{vals[1]:.2f}' + r', \ \dots, \ '
         + f'{vals[2]:.2f}'
         + r' \end{bmatrix}')
    e = Tex(s)
    e.set_color(color)
    e.scale(scale)
    e.move_to(position)
    return e

class P78_80(InteractiveScene):
    def construct(self):

        svgs_to_skip=[0, 1, 2]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        self.frame.reorient(0, 0, 0, (-0.1, 0.75, 0.0), 6.92)
        self.add(all_svgs[0], all_svgs[1], all_svgs[2], all_svgs[3],
                 all_svgs[4], all_svgs[5])

        self.wait()
        self.play(Write(all_svgs[6]), run_time=5)


        # Ok getting close here, goog next step is probably to go
        # figure out the matpotlib graph stuff
        # Hmm do I want to actually tackon/render in premier I guess?
        # I do think that would be easier. 
        # Ok improtant thing for now then is figuring out a reasonable
        # set of curves in matplotlib!
        # Ok done! Now bring in pictures. 

        imgs=Group()
        NUM_FRAMES=124
        for i in range(NUM_FRAMES):
            imgs.add(ImageMobject(str(img_dir+'/VJEPA_AC_CLIPS'+str(i).zfill(3)+'.jpg')))
        
        imgs_left=imgs.copy()
        imgs_right=imgs.copy()

        imgs[0].scale(0.42)
        imgs[0].move_to([-3.2, -1.7, 0])
        imgs[0].set_opacity(0.5)
        imgs[1].scale(0.42)
        imgs[1].move_to([-3.1, -1.8, 0])
        imgs[1].set_opacity(0.5)
        imgs[2].scale(0.42)
        imgs[2].move_to([-3.0, -1.9, 0])

        latest_img_copy=imgs[3].copy()
        imgs[3].scale(0.47)
        imgs[3].move_to([2.45, -1.77, 0])


        self.wait()
        #Bring in actuator signals first. 
        self.remove(all_svgs[6][1:])
        self.play(Write(all_svgs[8][1:]),
                 all_svgs[6][0].animate.move_to([-0.233, 0.95, 0]),
                 run_time=3)

        self.play(self.frame.animate.reorient(0, 0, 0, (-0.19, 0.27, 0.0), 7.43),
                  FadeOut(all_svgs[1]),
                  run_time=2)

        #Stack that videer
        self.add(imgs[0])
        self.wait(0.2)
        self.add(imgs[1])
        self.wait(0.2)
        self.add(imgs[2])
        self.wait(0.2)
        self.add(imgs[3])
        self.add(all_svgs[7])

    
        for im in imgs_left:
            im.scale(0.42).move_to([-3.0, -1.9, 0])

        for im in imgs_right:
            im.scale(0.47).move_to([2.45, -1.77, 0])

        self.wait()
        self.remove(imgs[2],imgs[3])
        for i in range(4, NUM_FRAMES):
            self.remove(imgs_left[i-2], imgs_right[i-1])
            self.add(imgs_left[i-1], imgs_right[i])
            self.wait(0.1)

        # for i in range(4, NUM_FRAMES):
        #     # promote imgs[i-1] from right (0.47) to front-left (0.42)
        #     imgs[i-1].scale(0.42 / 0.47).move_to([-3.0, -1.9, 0])
        #     # evict the old front-left
        #     self.remove(imgs[i-2])
        #     # bring in the new most-recent on the right
        #     imgs[i].scale(0.47).move_to([2.45, -1.77, 0])
        #     self.add(imgs[i])
        #     self.wait(0.1)

        self.play(FadeOut(imgs[0]),
                  FadeOut(imgs[1]),
                  FadeOut(imgs_left[i-1]), 
                  FadeOut(imgs_right[i]),
                  FadeOut(all_svgs[7]),
                  run_time=2)

        all_svgs[10].move_to([-0.15, -2.9, 0])
        self.play(FadeIn(imgs_right[i]), 
                  Write(all_svgs[10][5:]),
                  FadeIn(imgs[0]),
                  FadeIn(imgs[1]),
                  FadeIn(imgs[2]),
                  Write(all_svgs[10][:5]),
                  run_time=2)


        embedding_1=Tex(r'\begin{bmatrix} 0.22, \ -0.13, \ \dots, \ 0.31 \end{bmatrix}')
        embedding_1.set_color(BLUE)
        embedding_1.scale(0.5)
        embedding_1.move_to([2.3, 2.0, 0])
        embedding_1_og=embedding_1.copy()

        embedding_2=Tex(r'\begin{bmatrix} 0.05, \ -0.02, \ \dots, \ 0.50 \end{bmatrix}')
        embedding_2.set_color(YELLOW)
        embedding_2.scale(0.5)
        embedding_2.move_to([2.3, 1.4, 0])

        self.wait()
        self.play(FadeOut(all_svgs[5][1:]))
        self.play(Write(embedding_2),
                  FadeIn(all_svgs[11][:13]),
                  run_time=2)

        self.wait()
        all_svgs[11][13:].set_color(BLUE)
        self.play(Write(embedding_1),
                  FadeIn(all_svgs[11][13:]),
                  Write(all_svgs[9]),
                  run_time=2)

        # Ok so now play predicted embedding moving, 
        # Then one more robot sequence I think to play while Yann is 
        # talking!

        num_steps=NUM_FRAMES
        s1=np.linspace(0.22, 0.05, num_steps)
        s2=np.linspace(-0.13, -0.02, num_steps)
        s3=np.linspace(0.31, 0.5, num_steps)

        all_svgs[12].move_to([-5.4, -3.15, 0])
        self.wait()
        self.add(all_svgs[12])
        for n1, n2, n3 in zip(s1, s2, s3):
            # embedding_1=Tex(r'\begin{bmatrix} '+str(n1)+', \ -'+str(n2)+', \ \dots, \ '+str(n3)+' \end{bmatrix}')
            # embedding_1.set_color(BLUE)
            # embedding_1.scale(0.5)
            # embedding_1.move_to([2.3, 2.0, 0])

            self.remove(embedding_1)
            embedding_1=make_emb_tex([n1, n2, n3], color=BLUE, position=[2.3, 2.0, 0], scale=0.5)
            self.add(embedding_1)
            self.wait(0.1)


        self.wait()
        self.remove(imgs[2])
        self.remove(all_svgs[12])
        self.remove(embedding_1)
        self.add(embedding_1_og)
        self.wait()
        self.remove(embedding_1_og)
        for i in range(4, NUM_FRAMES):
            self.remove(imgs_left[i-2])
            self.add(imgs_left[i-1])
            self.remove(embedding_1)
            embedding_1=make_emb_tex([s1[i], s2[i], s3[i]], color=BLUE, position=[2.3, 2.0, 0], scale=0.5)
            self.add(embedding_1)
            self.wait(0.1)





        self.embed()
        self.wait(20)













