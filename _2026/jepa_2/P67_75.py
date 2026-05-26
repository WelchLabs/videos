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

def color_keyboard(action, keyboard_fill_svg):
    if action[0]>0.1:
        keyboard_fill_svg[2].set_opacity(np.max([0.2, float(abs(action[0]))])) #Right
        keyboard_fill_svg[0].set_opacity(0.0) #Left
    elif action[0]<-0.1:
        keyboard_fill_svg[0].set_opacity(np.max([0.2, float(abs(action[0]))])) 
        keyboard_fill_svg[2].set_opacity(0.0) 
    else:
        keyboard_fill_svg[0].set_opacity(0.0)
        keyboard_fill_svg[2].set_opacity(0.0)             

    if action[1]<-0.1:
        keyboard_fill_svg[3].set_opacity(np.max([0.2, float(abs(action[1]))])) #Right
        keyboard_fill_svg[1].set_opacity(0.0) 
    elif action[1]>0.1:
        keyboard_fill_svg[1].set_opacity(np.max([0.2, float(abs(action[1]))])) 
        keyboard_fill_svg[3].set_opacity(0.0) 
    else:
        keyboard_fill_svg[1].set_opacity(0.0)
        keyboard_fill_svg[3].set_opacity(0.0)     

svg_dir=Path('/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics/p67_76_to_manim')
img_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/graphics'
push_t_dir_1='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/hackin/push_t_episodes'
hackin_dir='/Users/stephen/Stephencwelch Dropbox/welch_labs/jepa_2/hackin'

class p69_75(InteractiveScene):
    def construct(self):


        svgs_to_skip=[0]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])


        p69_start=ImageMobject(str(hackin_dir+'/p69/start.png'))
        p69_left=ImageMobject(str(hackin_dir+'/p69/move_left.png'))
        p69_right=ImageMobject(str(hackin_dir+'/p69/move_right.png'))
        p69_up=ImageMobject(str(hackin_dir+'/p69/move_up.png'))
        p69_down=ImageMobject(str(hackin_dir+'/p69/move_down.png'))


        seed=2
        np.random.seed(seed)
        values_1 = np.clip(np.random.normal(0.5, 0.2, size=5), 0.0, 0.94)

        s = f"[ {values_1[0]:.1f}, {values_1[1]:.1f}, ... , {values_1[-1]:.1f} ]"
        emb_vector_1 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_1.set_color(FRESH_TAN)
        emb_vector_1.move_to([-3.35, 2.5, 0])

        values_2=values_1+np.random.randn(5)/12
        values_3=values_2+np.random.randn(5)/12

        s = f"[ {values_2[0]:.1f}, {values_2[1]:.1f}, ... , {values_2[-1]:.1f} ]"
        emb_vector_2 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_2.set_color(FRESH_TAN)
        emb_vector_2.move_to([3.35, 2.55, 0])


        p69_start.scale(0.7)
        p69_start.move_to([-3.4, -1.95, 0])

        for im in [p69_left, p69_right, p69_up, p69_down]:
            im.scale(0.7)
            im.move_to([3.4, -1.95, 0])


        self.wait()

        #Kinda thinking i can just cross-fade to this in editing?

        self.add(all_svgs[2], all_svgs[4][:-2], all_svgs[5])
        # self.add(emb_vector_1, emb_vector_2)

        self.add(all_svgs[10]) #Start position frame
        self.add(all_svgs[12]) #Predictor out arrow

        # self.remove(all_svgs[4][:-2])

        self.wait()
        self.add(p69_start)
        self.remove(all_svgs[10]); self.add(all_svgs[10])

        self.wait()
        self.play(Write(emb_vector_1), run_time=2)

        color_keyboard([0.0, -0.8], keyboard_fill_svg=all_svgs[6])
        self.wait()
        self.play(FadeIn(all_svgs[6]))


        self.wait()

        # all_svgs[6][3].set_opacity(0)


        # color_keyboard(actions_scaled[0], keyboard_fill_svg=all_svgs[6])

        # self.wait()
        # self.play(Write(all_svgs[2]), Write(all_svgs[3]), Write(all_svgs[4]),
        #           Write(emb_vector_1), Write(emb_vector_2), Write(emb_vector_3),
        #           Write(all_svgs[7]),
        #           run_time=4
        #           )

        # self.play(FadeIn(episode_1_imgs[0]), FadeIn(episode_1_imgs[1]),
        #           Write(all_svgs[0]), Write(all_svgs[1]),
        #           run_time=3)

        # self.play(Write(all_svgs[5]), run_time=2)
        # self.play(FadeIn(all_svgs[6]))




        # Pre-scale frames 2..N (0 and 1 already scaled above)
        for i in range(2, len(episode_1_imgs)):
            episode_1_imgs[i].scale(0.7)


        self.wait()




        self.wait(20)
        self.embed()



class p67(InteractiveScene):
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
        for i, p in enumerate(sorted((Path(push_t_dir_1)/'ep_0055/frames').glob('*.png'))):
            episode_1_imgs.add(ImageMobject(str(p)))

        episode_1_np=np.load(Path(push_t_dir_1)/'ep_0055/data.npz')
        episode_1_np['action'] #125,2


        #Hmm don't see embeddings, might just stick with random walk for these
        actions_scaled=episode_1_np['action']/np.max(np.abs(episode_1_np['action']), 0)
        
        # self.add(all_svgs[6][0]) #Left Fill 
        # self.add(all_svgs[6][1]) #Down
        # self.add(all_svgs[6][2]) #Right
        # self.add(all_svgs[6][3]) #Up

        all_svgs[6].set_opacity(0.0)
        # self.add(all_svgs[5]) #Keyboard
        # self.add(all_svgs[6]) #Keyboard fills 
        # self.add(all_svgs[0]) #Left frame
        # self.add(all_svgs[1]) #Right frame
        # self.add(all_svgs[2]) #X encoder
        # self.add(all_svgs[3]) #Y encoder
        # self.add(all_svgs[4]) #predictor
        # self.add(all_svgs[7])


        seed=2
        np.random.seed(seed)
        values_1 = np.clip(np.random.normal(0.5, 0.2, size=5), 0.0, 0.94)

        s = f"[ {values_1[0]:.1f}, {values_1[1]:.1f}, ... , {values_1[-1]:.1f} ]"
        emb_vector_1 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_1.set_color(FRESH_TAN)
        emb_vector_1.move_to([-3.35, 2.5, 0])

        values_2=values_1+np.random.randn(5)/12
        values_3=values_2+np.random.randn(5)/12

        s = f"[ {values_2[0]:.1f}, {values_2[1]:.1f}, ... , {values_2[-1]:.1f} ]"
        emb_vector_2 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_2.set_color(FRESH_TAN)
        emb_vector_2.move_to([3.4, 2.7, 0])

        s = f"[ {values_3[0]:.1f}, {values_3[1]:.1f}, ... , {values_3[-1]:.1f} ]"
        emb_vector_3 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_3.set_color(FRESH_TAN)
        emb_vector_3.move_to([3.4, 2.1, 0])

        # self.add(emb_vector_1, emb_vector_2, emb_vector_3)
        episode_1_imgs[0].scale(0.7)
        episode_1_imgs[0].move_to([-3.4, -1.95, 0])

        episode_1_imgs[1].scale(0.7)
        episode_1_imgs[1].move_to([3.4, -1.95, 0])

        color_keyboard(actions_scaled[0], keyboard_fill_svg=all_svgs[6])

        self.wait()
        self.play(Write(all_svgs[2]), Write(all_svgs[3]), Write(all_svgs[4]),
                  Write(emb_vector_1), Write(emb_vector_2), Write(emb_vector_3),
                  Write(all_svgs[7]),
                  run_time=4
                  )

        self.play(FadeIn(episode_1_imgs[0]), FadeIn(episode_1_imgs[1]),
                  Write(all_svgs[0]), Write(all_svgs[1]),
                  run_time=3)

        self.play(Write(all_svgs[5]), run_time=2)
        self.play(FadeIn(all_svgs[6]))




        # Pre-scale frames 2..N (0 and 1 already scaled above)
        for i in range(2, len(episode_1_imgs)):
            episode_1_imgs[i].scale(0.7)

        # Track running random-walk state for emb_vector_1
        current_values_1 = values_1.copy()

        # Playback parameters
        fps = 15
        dt = 1.0 / fps
        walk_step = 1.0 / 25   # std of random walk per frame for emb_1
        pert_step = 1.0 / 12   # perturbation magnitude for emb_2 / emb_3 (matches your original)

        # Frame 0 / frame 1 are already on screen — start at i=1 so pair (i, i+1) = (1, 2)
        prev_left = episode_1_imgs[0]
        prev_right = episode_1_imgs[1]

        n_steps = min(len(episode_1_imgs) - 1, len(actions_scaled))
        # n_steps=20

        self.wait()
        for i in tqdm(range(1, n_steps)):
            # --- swap frames ---
            self.remove(prev_left, prev_right)
            new_left = episode_1_imgs[i]
            new_right = episode_1_imgs[i + 1]
            new_left.move_to([-3.4, -1.95, 0])
            new_right.move_to([3.4, -1.95, 0])
            self.add(new_left, new_right)

            # keep SVG borders / keyboard on top of the images
            for j in (0, 1, 5, 6):
                self.remove(all_svgs[j]); self.add(all_svgs[j])

            prev_left, prev_right = new_left, new_right

            # --- update keyboard fills ---
            color_keyboard(actions_scaled[i], keyboard_fill_svg=all_svgs[6])

            # --- random walk on values_1, perturbations for values_2/3 ---
            current_values_1 = current_values_1 + np.random.randn(5) * walk_step
            current_values_1 = np.clip(current_values_1, 0.0, 0.94)

            v2 = np.clip(current_values_1 + np.random.randn(5) * pert_step, 0.0, 0.94)
            v3 = np.clip(v2 + np.random.randn(5) * pert_step, 0.0, 0.94)

            # --- rebuild Text objects (cheap enough at 125 frames) ---
            new_emb_1 = Text(
                f"[ {current_values_1[0]:.1f}, {current_values_1[1]:.1f}, ... , {current_values_1[-1]:.1f} ]",
                color=FRESH_TAN, font_size=35,
            ).move_to([-3.35, 2.5, 0])

            new_emb_2 = Text(
                f"[ {v2[0]:.1f}, {v2[1]:.1f}, ... , {v2[-1]:.1f} ]",
                color=FRESH_TAN, font_size=35,
            ).move_to([3.4, 2.7, 0])

            new_emb_3 = Text(
                f"[ {v3[0]:.1f}, {v3[1]:.1f}, ... , {v3[-1]:.1f} ]",
                color=FRESH_TAN, font_size=35,
            ).move_to([3.4, 2.1, 0])

            self.remove(emb_vector_1, emb_vector_2, emb_vector_3)
            emb_vector_1, emb_vector_2, emb_vector_3 = new_emb_1, new_emb_2, new_emb_3
            self.add(emb_vector_1, emb_vector_2, emb_vector_3)

            self.wait(dt)


        # self.add(episode_1_imgs[0])
        # self.add(episode_1_imgs[1])
        # self.remove(all_svgs[0]); self.add(all_svgs[0])  #Left frame
        # self.remove(all_svgs[1]); self.add(all_svgs[1])  #Right frame

    

        #Ok, that looks good now I gotta figure out how to lerps it

        self.wait()
        self.embed(20)

class p67b(InteractiveScene):
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
        for i, p in enumerate(sorted((Path(push_t_dir_1)/'ep_0070/frames').glob('*.png'))):
            episode_1_imgs.add(ImageMobject(str(p)))

        episode_1_np=np.load(Path(push_t_dir_1)/'ep_0055/data.npz')
        episode_1_np['action'] #125,2

        #Hmm don't see embeddings, might just stick with random walk for these
        actions_scaled=episode_1_np['action']/np.max(np.abs(episode_1_np['action']), 0)
        
        # self.add(all_svgs[6][0]) #Left Fill 
        # self.add(all_svgs[6][1]) #Down
        # self.add(all_svgs[6][2]) #Right
        # self.add(all_svgs[6][3]) #Up

        all_svgs[6].set_opacity(0.0)
        # self.add(all_svgs[5]) #Keyboard
        # self.add(all_svgs[6]) #Keyboard fills 
        # self.add(all_svgs[0]) #Left frame
        # self.add(all_svgs[1]) #Right frame
        # self.add(all_svgs[2]) #X encoder
        # self.add(all_svgs[3]) #Y encoder
        # self.add(all_svgs[4]) #predictor
        # self.add(all_svgs[7])


        seed=2
        np.random.seed(seed)
        values_1 = np.clip(np.random.normal(0.5, 0.2, size=5), 0.0, 0.94)

        s = f"[ {values_1[0]:.1f}, {values_1[1]:.1f}, ... , {values_1[-1]:.1f} ]"
        emb_vector_1 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_1.set_color(FRESH_TAN)
        emb_vector_1.move_to([-3.35, 2.5, 0])

        values_2=values_1+np.random.randn(5)/12
        values_3=values_2+np.random.randn(5)/12

        s = f"[ {values_2[0]:.1f}, {values_2[1]:.1f}, ... , {values_2[-1]:.1f} ]"
        emb_vector_2 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_2.set_color(FRESH_TAN)
        emb_vector_2.move_to([3.4, 2.7, 0])

        s = f"[ {values_3[0]:.1f}, {values_3[1]:.1f}, ... , {values_3[-1]:.1f} ]"
        emb_vector_3 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_3.set_color(FRESH_TAN)
        emb_vector_3.move_to([3.4, 2.1, 0])

        # self.add(emb_vector_1, emb_vector_2, emb_vector_3)
        episode_1_imgs[0].scale(0.7)
        episode_1_imgs[0].move_to([-3.4, -1.95, 0])

        episode_1_imgs[1].scale(0.7)
        episode_1_imgs[1].move_to([3.4, -1.95, 0])

        color_keyboard(actions_scaled[0], keyboard_fill_svg=all_svgs[6])

        self.wait()
        self.play(Write(all_svgs[2]), Write(all_svgs[3]), Write(all_svgs[4]),
                  Write(emb_vector_1), Write(emb_vector_2), Write(emb_vector_3),
                  Write(all_svgs[7]),
                  run_time=4
                  )

        self.play(FadeIn(episode_1_imgs[0]), FadeIn(episode_1_imgs[1]),
                  Write(all_svgs[0]), Write(all_svgs[1]),
                  run_time=3)

        self.play(Write(all_svgs[5]), run_time=2)
        self.play(FadeIn(all_svgs[6]))




        # Pre-scale frames 2..N (0 and 1 already scaled above)
        for i in range(2, len(episode_1_imgs)):
            episode_1_imgs[i].scale(0.7)

        # Track running random-walk state for emb_vector_1
        current_values_1 = values_1.copy()

        # Playback parameters
        fps = 15
        dt = 1.0 / fps
        walk_step = 1.0 / 25   # std of random walk per frame for emb_1
        pert_step = 1.0 / 12   # perturbation magnitude for emb_2 / emb_3 (matches your original)

        # Frame 0 / frame 1 are already on screen — start at i=1 so pair (i, i+1) = (1, 2)
        prev_left = episode_1_imgs[0]
        prev_right = episode_1_imgs[1]

        n_steps = min(len(episode_1_imgs) - 1, len(actions_scaled))
        # n_steps=20

        self.wait()
        for i in tqdm(range(1, n_steps)):
            # --- swap frames ---
            self.remove(prev_left, prev_right)
            new_left = episode_1_imgs[i]
            new_right = episode_1_imgs[i + 1]
            new_left.move_to([-3.4, -1.95, 0])
            new_right.move_to([3.4, -1.95, 0])
            self.add(new_left, new_right)

            # keep SVG borders / keyboard on top of the images
            for j in (0, 1, 5, 6):
                self.remove(all_svgs[j]); self.add(all_svgs[j])

            prev_left, prev_right = new_left, new_right

            # --- update keyboard fills ---
            color_keyboard(actions_scaled[i], keyboard_fill_svg=all_svgs[6])

            # --- random walk on values_1, perturbations for values_2/3 ---
            current_values_1 = current_values_1 + np.random.randn(5) * walk_step
            current_values_1 = np.clip(current_values_1, 0.0, 0.94)

            v2 = np.clip(current_values_1 + np.random.randn(5) * pert_step, 0.0, 0.94)
            v3 = np.clip(v2 + np.random.randn(5) * pert_step, 0.0, 0.94)

            # --- rebuild Text objects (cheap enough at 125 frames) ---
            new_emb_1 = Text(
                f"[ {current_values_1[0]:.1f}, {current_values_1[1]:.1f}, ... , {current_values_1[-1]:.1f} ]",
                color=FRESH_TAN, font_size=35,
            ).move_to([-3.35, 2.5, 0])

            new_emb_2 = Text(
                f"[ {v2[0]:.1f}, {v2[1]:.1f}, ... , {v2[-1]:.1f} ]",
                color=FRESH_TAN, font_size=35,
            ).move_to([3.4, 2.7, 0])

            new_emb_3 = Text(
                f"[ {v3[0]:.1f}, {v3[1]:.1f}, ... , {v3[-1]:.1f} ]",
                color=FRESH_TAN, font_size=35,
            ).move_to([3.4, 2.1, 0])

            self.remove(emb_vector_1, emb_vector_2, emb_vector_3)
            emb_vector_1, emb_vector_2, emb_vector_3 = new_emb_1, new_emb_2, new_emb_3
            self.add(emb_vector_1, emb_vector_2, emb_vector_3)

            self.wait(dt)


        # self.add(episode_1_imgs[0])
        # self.add(episode_1_imgs[1])
        # self.remove(all_svgs[0]); self.add(all_svgs[0])  #Left frame
        # self.remove(all_svgs[1]); self.add(all_svgs[1])  #Right frame

    

        #Ok, that looks good now I gotta figure out how to lerps it

        self.wait()
        self.embed(20)

class p67c(InteractiveScene):
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
        for i, p in enumerate(sorted((Path(push_t_dir_1)/'ep_0060/frames').glob('*.png'))):
            episode_1_imgs.add(ImageMobject(str(p)))

        episode_1_np=np.load(Path(push_t_dir_1)/'ep_0055/data.npz')
        episode_1_np['action'] #125,2 

        #Hmm don't see embeddings, might just stick with random walk for these
        actions_scaled=episode_1_np['action']/np.max(np.abs(episode_1_np['action']), 0)
        
        # self.add(all_svgs[6][0]) #Left Fill 
        # self.add(all_svgs[6][1]) #Down
        # self.add(all_svgs[6][2]) #Right
        # self.add(all_svgs[6][3]) #Up

        all_svgs[6].set_opacity(0.0)
        # self.add(all_svgs[5]) #Keyboard
        # self.add(all_svgs[6]) #Keyboard fills 
        # self.add(all_svgs[0]) #Left frame
        # self.add(all_svgs[1]) #Right frame
        # self.add(all_svgs[2]) #X encoder
        # self.add(all_svgs[3]) #Y encoder
        # self.add(all_svgs[4]) #predictor
        # self.add(all_svgs[7])


        seed=2
        np.random.seed(seed)
        values_1 = np.clip(np.random.normal(0.5, 0.2, size=5), 0.0, 0.94)

        s = f"[ {values_1[0]:.1f}, {values_1[1]:.1f}, ... , {values_1[-1]:.1f} ]"
        emb_vector_1 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_1.set_color(FRESH_TAN)
        emb_vector_1.move_to([-3.35, 2.5, 0])

        values_2=values_1+np.random.randn(5)/12
        values_3=values_2+np.random.randn(5)/12

        s = f"[ {values_2[0]:.1f}, {values_2[1]:.1f}, ... , {values_2[-1]:.1f} ]"
        emb_vector_2 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_2.set_color(FRESH_TAN)
        emb_vector_2.move_to([3.4, 2.7, 0])

        s = f"[ {values_3[0]:.1f}, {values_3[1]:.1f}, ... , {values_3[-1]:.1f} ]"
        emb_vector_3 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_3.set_color(FRESH_TAN)
        emb_vector_3.move_to([3.4, 2.1, 0])

        # self.add(emb_vector_1, emb_vector_2, emb_vector_3)
        episode_1_imgs[0].scale(0.7)
        episode_1_imgs[0].move_to([-3.4, -1.95, 0])

        episode_1_imgs[1].scale(0.7)
        episode_1_imgs[1].move_to([3.4, -1.95, 0])

        color_keyboard(actions_scaled[0], keyboard_fill_svg=all_svgs[6])

        self.wait()
        self.play(Write(all_svgs[2]), Write(all_svgs[3]), Write(all_svgs[4]),
                  Write(emb_vector_1), Write(emb_vector_2), Write(emb_vector_3),
                  Write(all_svgs[7]),
                  run_time=4
                  )

        self.play(FadeIn(episode_1_imgs[0]), FadeIn(episode_1_imgs[1]),
                  Write(all_svgs[0]), Write(all_svgs[1]),
                  run_time=3)

        self.play(Write(all_svgs[5]), run_time=2)
        self.play(FadeIn(all_svgs[6]))




        # Pre-scale frames 2..N (0 and 1 already scaled above)
        for i in range(2, len(episode_1_imgs)):
            episode_1_imgs[i].scale(0.7)

        # Track running random-walk state for emb_vector_1
        current_values_1 = values_1.copy()

        # Playback parameters
        fps = 15
        dt = 1.0 / fps
        walk_step = 1.0 / 25   # std of random walk per frame for emb_1
        pert_step = 1.0 / 12   # perturbation magnitude for emb_2 / emb_3 (matches your original)

        # Frame 0 / frame 1 are already on screen — start at i=1 so pair (i, i+1) = (1, 2)
        prev_left = episode_1_imgs[0]
        prev_right = episode_1_imgs[1]

        n_steps = min(len(episode_1_imgs) - 1, len(actions_scaled))
        # n_steps=20

        self.wait()
        for i in tqdm(range(1, n_steps)):
            # --- swap frames ---
            self.remove(prev_left, prev_right)
            new_left = episode_1_imgs[i]
            new_right = episode_1_imgs[i + 1]
            new_left.move_to([-3.4, -1.95, 0])
            new_right.move_to([3.4, -1.95, 0])
            self.add(new_left, new_right)

            # keep SVG borders / keyboard on top of the images
            for j in (0, 1, 5, 6):
                self.remove(all_svgs[j]); self.add(all_svgs[j])

            prev_left, prev_right = new_left, new_right

            # --- update keyboard fills ---
            color_keyboard(actions_scaled[i], keyboard_fill_svg=all_svgs[6])

            # --- random walk on values_1, perturbations for values_2/3 ---
            current_values_1 = current_values_1 + np.random.randn(5) * walk_step
            current_values_1 = np.clip(current_values_1, 0.0, 0.94)

            v2 = np.clip(current_values_1 + np.random.randn(5) * pert_step, 0.0, 0.94)
            v3 = np.clip(v2 + np.random.randn(5) * pert_step, 0.0, 0.94)

            # --- rebuild Text objects (cheap enough at 125 frames) ---
            new_emb_1 = Text(
                f"[ {current_values_1[0]:.1f}, {current_values_1[1]:.1f}, ... , {current_values_1[-1]:.1f} ]",
                color=FRESH_TAN, font_size=35,
            ).move_to([-3.35, 2.5, 0])

            new_emb_2 = Text(
                f"[ {v2[0]:.1f}, {v2[1]:.1f}, ... , {v2[-1]:.1f} ]",
                color=FRESH_TAN, font_size=35,
            ).move_to([3.4, 2.7, 0])

            new_emb_3 = Text(
                f"[ {v3[0]:.1f}, {v3[1]:.1f}, ... , {v3[-1]:.1f} ]",
                color=FRESH_TAN, font_size=35,
            ).move_to([3.4, 2.1, 0])

            self.remove(emb_vector_1, emb_vector_2, emb_vector_3)
            emb_vector_1, emb_vector_2, emb_vector_3 = new_emb_1, new_emb_2, new_emb_3
            self.add(emb_vector_1, emb_vector_2, emb_vector_3)

            self.wait(dt)


        # self.add(episode_1_imgs[0])
        # self.add(episode_1_imgs[1])
        # self.remove(all_svgs[0]); self.add(all_svgs[0])  #Left frame
        # self.remove(all_svgs[1]); self.add(all_svgs[1])  #Right frame

    

        #Ok, that looks good now I gotta figure out how to lerps it

        self.wait()
        self.embed(20)



# class p67_75_setup_test(InteractiveScene):
#     def construct(self):

#         svgs_to_skip=[0]
#         svg_files=list(sorted(svg_dir.glob('*.svg')))
#         all_svgs=Group()
#         for i, svg_file in enumerate(svg_files): 
#             if i in svgs_to_skip: continue
#             svg_image=SVGMobject(str(svg_file))
#             svg_image.scale(4.0)
#             all_svgs.add(svg_image[1:])

        
#         episode_1_imgs=Group()
#         for i, p in enumerate((Path(push_t_dir_1)/'ep_0055/frames').glob('*.png')):
#             episode_1_imgs.add(ImageMobject(str(p)))

#         episode_1_np=np.load(Path(push_t_dir_1)/'ep_0055/data.npz')
#         episode_1_np['action'] #125,2

#         def color_keyboard(action):
#             if action[0]>=0:
#                 all_svgs[6][2].set_opacity(np.max([0.4, float(abs(action[0]))])) #Right
#                 all_svgs[6][0].set_opacity(0.0) #Left
#             else:
#                 all_svgs[6][0].set_opacity(np.max([0.4, float(abs(action[0]))])) 
#                 all_svgs[6][2].set_opacity(0.0) 

#             if action[1]>=0:
#                 all_svgs[6][3].set_opacity(np.max([0.4, float(abs(action[1]))])) #Right
#                 all_svgs[6][1].set_opacity(0.0) #Left
#             else:
#                 all_svgs[6][1].set_opacity(np.max([0.4, float(abs(action[1]))])) 
#                 all_svgs[6][3].set_opacity(0.0) 


#         #Hmm don't see embeddings, might just stick with random walk for these
#         actions_scaled=episode_1_np['action']/np.max(np.abs(episode_1_np['action']), 0)
        
#         # self.add(all_svgs[6][0]) #Left Fill 
#         # self.add(all_svgs[6][1]) #Down
#         # self.add(all_svgs[6][2]) #Right
#         # self.add(all_svgs[6][3]) #Up


#         self.add(all_svgs[5]) #Keyboard
#         all_svgs[6].set_opacity(0.0)
#         self.add(all_svgs[6]) #Keyboard fills 

#         self.add(all_svgs[0]) #Left frame
#         self.add(all_svgs[1]) #Right frame
#         self.add(all_svgs[2]) #X encoder
#         self.add(all_svgs[3]) #Y encoder
#         self.add(all_svgs[4]) #predictor
#         self.add(all_svgs[7])

#         #Hmm ok ok ok ok ok ok now we need some vectors. 

#         seed=2
#         np.random.seed(seed)
#         values_1 = np.clip(np.random.normal(0.5, 0.2, size=5), 0.0, 0.94)

#         s = f"[ {values_1[0]:.1f}, {values_1[1]:.1f}, ... , {values_1[-1]:.1f} ]"
#         emb_vector_1 = Text(s, color=FRESH_TAN, font_size=35)
#         emb_vector_1.set_color(FRESH_TAN)
#         emb_vector_1.move_to([-3.35, 2.5, 0])

#         values_2=values_1+np.random.randn(5)/12
#         values_3=values_2+np.random.randn(5)/12

#         s = f"[ {values_2[0]:.1f}, {values_2[1]:.1f}, ... , {values_2[-1]:.1f} ]"
#         emb_vector_2 = Text(s, color=FRESH_TAN, font_size=35)
#         emb_vector_2.set_color(FRESH_TAN)
#         emb_vector_2.move_to([3.4, 2.7, 0])

#         s = f"[ {values_3[0]:.1f}, {values_3[1]:.1f}, ... , {values_3[-1]:.1f} ]"
#         emb_vector_3 = Text(s, color=FRESH_TAN, font_size=35)
#         emb_vector_3.set_color(FRESH_TAN)
#         emb_vector_3.move_to([3.4, 2.1, 0])

#         self.add(emb_vector_1, emb_vector_2, emb_vector_3)

#         #Ok, that looks good now I gotta figure out how to lerps it

#         color_keyboard(actions_scaled[2])

#         episode_1_imgs[0].scale(0.7)
#         episode_1_imgs[0].move_to([-3.4, -1.95, 0])


#         episode_1_imgs[1].scale(0.7)
#         episode_1_imgs[1].move_to([3.4, -1.95, 0])

#         self.add(episode_1_imgs[0])
#         self.add(episode_1_imgs[1])
#         self.remove(all_svgs[0]); self.add(all_svgs[0])  #Left frame
#         self.remove(all_svgs[1]); self.add(all_svgs[1])  #Right frame


#         self.wait()
#         self.embed(20)












