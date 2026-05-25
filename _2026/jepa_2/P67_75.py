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

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/p67_76_to_manim')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics'
push_t_dir_1='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/hackin/push_t_episodes'




class p67_75(InteractiveScene):
    def construct(self):

        svgs_to_skip=[0]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        
        episode_1_imgs=Group()
        for i, p in enumerate((Path(push_t_dir_1)/'ep_0055/frames').glob('*.png')):
            episode_1_imgs.add(ImageMobject(str(p)))

        episode_1_np=np.load(Path(push_t_dir_1)/'ep_0055/data.npz')
        episode_1_np['action'] #125,2

        def color_keyboard(action):
            if action[0]>=0:
                all_svgs[6][2].set_opacity(np.max([0.4, float(abs(action[0]))])) #Right
                all_svgs[6][0].set_opacity(0.0) #Left
            else:
                all_svgs[6][0].set_opacity(np.max([0.4, float(abs(action[0]))])) 
                all_svgs[6][2].set_opacity(0.0) 

            if action[1]>=0:
                all_svgs[6][3].set_opacity(np.max([0.4, float(abs(action[1]))])) #Right
                all_svgs[6][1].set_opacity(0.0) #Left
            else:
                all_svgs[6][1].set_opacity(np.max([0.4, float(abs(action[1]))])) 
                all_svgs[6][3].set_opacity(0.0) 


        #Hmm don't see embeddings, might just stick with random walk for these
        actions_scaled=episode_1_np['action']/np.max(np.abs(episode_1_np['action']), 0)
        
        # self.add(all_svgs[6][0]) #Left Fill 
        # self.add(all_svgs[6][1]) #Down
        # self.add(all_svgs[6][2]) #Right
        # self.add(all_svgs[6][3]) #Up


        self.add(all_svgs[5]) #Keyboard
        all_svgs[6].set_opacity(0.0)
        self.add(all_svgs[6]) #Keyboard fills 

        self.add(all_svgs[0]) #Left frame
        self.add(all_svgs[1]) #Right frame
        self.add(all_svgs[2]) #X encoder
        self.add(all_svgs[3]) #Y encoder
        self.add(all_svgs[4]) #predictor

        #Hmm ok ok ok ok ok ok now we need some vector
        

        color_keyboard(actions_scaled[2])




        self.add(episode_1_imgs[0])


        self.wait()
        self.embed(20)