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


        # p69_start=ImageMobject(str(hackin_dir+'/p69/start.png'))
        # p69_left=ImageMobject(str(hackin_dir+'/p69/move_left.png'))
        # p69_right=ImageMobject(str(hackin_dir+'/p69/move_right.png'))
        # p69_up=ImageMobject(str(hackin_dir+'/p69/move_up.png'))
        # p69_down=ImageMobject(str(hackin_dir+'/p69/move_down.png'))

        p69_start=ImageMobject(str(hackin_dir+'/p71/ep_18002/before.png'))
        p69_left=ImageMobject(str(hackin_dir+'/p71/ep_18002/left.png'))
        p69_right=ImageMobject(str(hackin_dir+'/p71/ep_18002/right.png'))
        p69_up=ImageMobject(str(hackin_dir+'/p71/ep_18002/up.png'))
        p69_down=ImageMobject(str(hackin_dir+'/p71/ep_18002/down.png'))

        seed=2
        np.random.seed(seed)
        values_1 = np.clip(np.random.normal(0.5, 0.2, size=5), 0.0, 0.94)

        s = f"[ {values_1[0]:.1f}, {values_1[1]:.1f}, ... , {values_1[-1]:.1f} ]"
        emb_vector_1 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_1.set_color(FRESH_TAN)
        emb_vector_1.move_to([-3.35, 2.5, 0])

        values_2=values_1+np.random.randn(5)/12

        s = f"[ {values_2[0]:.1f}, {values_2[1]:.1f}, ... , {values_2[-1]:.1f} ]"
        emb_vector_2 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_2.set_color(FRESH_TAN)
        emb_vector_2.move_to([3.35, 2.55, 0])


        p69_start.scale(0.7)
        p69_start.move_to([-3.4, -1.95, 0])

        p69_up.scale(0.7)
        p69_up.move_to([3.45, -1.95, 0])

        p69_down.scale(0.7)
        p69_down.move_to([3.45, -1.95, 0])

        p69_left.scale(0.7)
        p69_left.move_to([3.45, -1.95, 0])

        p69_right.scale(0.7)
        p69_right.move_to([3.45, -1.95, 0])

        all_svgs[9].shift([0, -0.07, 0])


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
        self.play(Write(emb_vector_2), run_time=3)

        self.wait()
        self.play(Write(all_svgs[11]),
                  Write(all_svgs[9]),
                  run_time=3)
        self.add(p69_up)
        self.remove(all_svgs[9]); self.add(all_svgs[9])


        #P70
        self.wait()
        self.remove(p69_up)
        self.add(p69_left)
        color_keyboard([-0.8, 0.0], keyboard_fill_svg=all_svgs[6])
        self.remove(all_svgs[9]); self.add(all_svgs[9])

        self.wait()
        self.remove(p69_left)
        self.add(p69_right)
        color_keyboard([0.8, 0.0], keyboard_fill_svg=all_svgs[6])
        self.remove(all_svgs[9]); self.add(all_svgs[9])

        self.wait()
        self.remove(p69_right)
        self.add(p69_down)
        color_keyboard([0.0, 0.8], keyboard_fill_svg=all_svgs[6])
        self.remove(all_svgs[9]); self.add(all_svgs[9])

        # P71 - Roll this sucker out!
        # Need to load in the real actions here
        # And change first to setp to the actualy first step, then
        # I can rollout to the next step! 
        episode_imgs=Group()
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_18002/wm/').glob('*.png'))):
            episode_imgs.add(ImageMobject(str(p)))

        episode_imgs[1].scale(0.7)
        episode_imgs[1].move_to([3.45, -1.95, 0])

        step=0
        ep_data=np.load(hackin_dir+'/p71/ep_18002/wm/data.npz')
        step_actions=ep_data['action_blocks'][step].reshape(5,2) #Not sure if this is the right reshape
        aggregate_actions=step_actions.mean(0)

        self.wait()
        color_keyboard(aggregate_actions, keyboard_fill_svg=all_svgs[6])
        self.remove(p69_down)
        self.add(episode_imgs[1])
        self.remove(all_svgs[9]); self.add(all_svgs[9])


        self.wait()

        # self.add(all_svgs[4])

        step=1
        horizontal_offset=6.7

        #First follout step:
        predictor_copy=all_svgs[4][:-2].copy()
        predictor_out_arrow_copy=all_svgs[12].copy()
        decoder_copy=all_svgs[11].copy()
        next_frame_copy=all_svgs[9].copy()
        keyboard_copy=all_svgs[5].copy()
        keyboard_fill_copy=all_svgs[6].copy()
        keyboard_fill_copy.set_opacity(0.0)

        values_3=values_2+np.random.randn(5)/12
        s = f"[ {values_3[0]:.1f}, {values_3[1]:.1f}, ... , {values_3[-1]:.1f} ]"
        emb_vector_3 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_3.set_color(FRESH_TAN)
        emb_vector_3.move_to([3.35, 2.55, 0])

        rollout_group=Group(predictor_copy, predictor_out_arrow_copy, decoder_copy, next_frame_copy,  keyboard_copy, keyboard_fill_copy, emb_vector_3)
        
        episode_imgs[step+1].scale(0.7)
        episode_imgs[step+1].move_to([3.45+horizontal_offset*(step), -1.95, 0])

        step_actions=ep_data['action_blocks'][step].reshape(5,2) #Not sure if this is the right reshape
        aggregate_actions=step_actions.mean(0)

        self.wait()
        self.add(rollout_group)
        self.play(self.frame.animate.reorient(0, 0, 0, (3.41, -0.07, 0.0), 10.30), 
                  rollout_group.animate.shift([horizontal_offset*(step), 0.0, 0]),
                  run_time=5
                  )
        color_keyboard(aggregate_actions, keyboard_fill_copy)
        self.add(episode_imgs[step+1])
        self.remove(next_frame_copy); self.add(next_frame_copy)

        
        step=2
        horizontal_offset=6.7

        predictor_copy_b=predictor_copy.copy()
        predictor_out_arrow_copy_b=predictor_out_arrow_copy.copy()
        decoder_copy_b=decoder_copy.copy()
        next_frame_copy_b=next_frame_copy.copy()
        keyboard_copy_b=keyboard_copy.copy()
        keyboard_fill_copy_b=keyboard_fill_copy.copy()
        keyboard_fill_copy_b.set_opacity(0.0)

        values_4=values_3+np.random.randn(5)/12
        s = f"[ {values_4[0]:.1f}, {values_4[1]:.1f}, ... , {values_4[-1]:.1f} ]"
        emb_vector_4 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_4.set_color(FRESH_TAN)
        emb_vector_4.move_to([3.35+step*horizontal_offset, 2.55, 0])

        rollout_group=Group(predictor_copy_b, predictor_out_arrow_copy_b, decoder_copy_b, next_frame_copy_b,  keyboard_copy_b)

        episode_imgs[step+1].scale(0.7)
        episode_imgs[step+1].move_to([3.45+step*horizontal_offset, -1.95, 0])

        step_actions=ep_data['action_blocks'][step].reshape(5,2) #Not sure if this is the right reshape
        aggregate_actions=step_actions.mean(0)

        self.wait()
        self.add(rollout_group)
        color_keyboard(aggregate_actions, keyboard_fill_copy) #Opaicty

        self.play(self.frame.animate.reorient(0, 0, 0, (6.64, -0.08, 0.0), 14.45), 
                  rollout_group.animate.shift([horizontal_offset, 0.0, 0]),
                  run_time=5
                  )
        keyboard_fill_copy_b.shift([horizontal_offset, 0.0, 0])
        self.add(keyboard_fill_copy_b)
        color_keyboard(aggregate_actions, keyboard_fill_copy_b)
        self.add(emb_vector_4)
        self.add(episode_imgs[step+1])
        self.remove(next_frame_copy_b); self.add(next_frame_copy_b)

        #Ok last one I think
        step=3
        horizontal_offset=6.7

        predictor_copy_c=predictor_copy_b.copy()
        predictor_out_arrow_copy_c=predictor_out_arrow_copy_b.copy()
        decoder_copy_c=decoder_copy_b.copy()
        next_frame_copy_c=next_frame_copy_b.copy()
        keyboard_copy_c=keyboard_copy_b.copy()
        keyboard_fill_copy_c=keyboard_fill_copy_b.copy()
        keyboard_fill_copy_c.set_opacity(0.0)

        values_5=values_4+np.random.randn(5)/12
        s = f"[ {values_5[0]:.1f}, {values_5[1]:.1f}, ... , {values_5[-1]:.1f} ]"
        emb_vector_5 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_5.set_color(FRESH_TAN)
        emb_vector_5.move_to([3.35+step*horizontal_offset, 2.55, 0])

        rollout_group=Group(predictor_copy_c, predictor_out_arrow_copy_c, decoder_copy_c, next_frame_copy_c,  keyboard_copy_c)

        episode_imgs[step+1].scale(0.7)
        episode_imgs[step+1].move_to([3.45+step*horizontal_offset, -1.95, 0])

        step_actions=ep_data['action_blocks'][step].reshape(5,2) #Not sure if this is the right reshape
        aggregate_actions=step_actions.mean(0)

        self.wait()
        self.add(rollout_group)
        color_keyboard(aggregate_actions, keyboard_fill_copy) #Opaicty

        self.play(self.frame.animate.reorient(0, 0, 0, (9.97, -0.09, 0.0), 18.35),
                  rollout_group.animate.shift([horizontal_offset, 0.0, 0]),
                  run_time=5
                  )
        keyboard_fill_copy_c.shift([horizontal_offset, 0.0, 0])
        self.add(keyboard_fill_copy_c)
        color_keyboard(aggregate_actions, keyboard_fill_copy_c)
        self.add(emb_vector_5)
        self.add(episode_imgs[step+1])
        self.remove(next_frame_copy_c); self.add(next_frame_copy_c)


        # Ok ok ok maybe I'm getting too crazy here, but I think i want to do one more
        # I think I'll have time with the VO
        step=4
        horizontal_offset=6.7

        predictor_copy_d=predictor_copy_c.copy()
        predictor_out_arrow_copy_d=predictor_out_arrow_copy_c.copy()
        decoder_copy_d=decoder_copy_c.copy()
        next_frame_copy_d=next_frame_copy_c.copy()
        keyboard_copy_d=keyboard_copy_c.copy()
        keyboard_fill_copy_d=keyboard_fill_copy_c.copy()
        keyboard_fill_copy_d.set_opacity(0.0)

        values_6=values_5+np.random.randn(5)/12
        s = f"[ {values_6[0]:.1f}, {values_6[1]:.1f}, ... , {values_6[-1]:.1f} ]"
        emb_vector_6 = Text(s, color=FRESH_TAN, font_size=35)
        emb_vector_6.set_color(FRESH_TAN)
        emb_vector_6.move_to([3.35+step*horizontal_offset, 2.55, 0])

        rollout_group=Group(predictor_copy_d, predictor_out_arrow_copy_d, decoder_copy_d, next_frame_copy_d,  keyboard_copy_d)

        episode_imgs[step+1].scale(0.7)
        episode_imgs[step+1].move_to([3.45+step*horizontal_offset, -1.95, 0])

        step_actions=ep_data['action_blocks'][step].reshape(5,2) #Not sure if this is the right reshape
        aggregate_actions=step_actions.mean(0)

        self.wait()
        self.add(rollout_group)
        color_keyboard(aggregate_actions, keyboard_fill_copy) #Opaicty

        self.play(self.frame.animate.reorient(0, 0, 0, (13.37, 0.08, 0.0), 21.66),
                  rollout_group.animate.shift([horizontal_offset, 0.0, 0]),
                  run_time=5
                  )
        keyboard_fill_copy_d.shift([horizontal_offset, 0.0, 0])
        self.add(keyboard_fill_copy_d)
        color_keyboard(aggregate_actions, keyboard_fill_copy_d)
        self.add(emb_vector_6)
        self.add(episode_imgs[step+1])
        self.remove(next_frame_copy_d); self.add(next_frame_copy_d)


        # P72 Ok so little chnange of plan here
        # i think i just want to bring these six frames togheter 
        # and play as a video. I think that will be more clear!

# Group everything currently on screen for removal before P72
        everything_on_screen = Group(
            # Original elements
            all_svgs[2],
            all_svgs[4][:-2],
            all_svgs[5],
            all_svgs[6],
            all_svgs[9][:-1],
            all_svgs[10][:-1],
            all_svgs[11],
            all_svgs[12],
            emb_vector_1,
            emb_vector_2,
            # p69_start,
            # episode_imgs[1],
            # Step 1 rollout
            predictor_copy,
            predictor_out_arrow_copy,
            decoder_copy,
            next_frame_copy[:-1],
            keyboard_copy,
            keyboard_fill_copy,
            emb_vector_3,
            # episode_imgs[2],
            # Step 2 rollout
            predictor_copy_b,
            predictor_out_arrow_copy_b,
            decoder_copy_b,
            next_frame_copy_b[:-1],
            keyboard_copy_b,
            keyboard_fill_copy_b,
            emb_vector_4,
            # episode_imgs[3],
            # Step 3 rollout
            predictor_copy_c,
            predictor_out_arrow_copy_c,
            decoder_copy_c,
            next_frame_copy_c[:-1],
            keyboard_copy_c,
            keyboard_fill_copy_c,
            emb_vector_5,
            # episode_imgs[4],

            # Step 4 rollout
            predictor_copy_d,
            predictor_out_arrow_copy_d,
            decoder_copy_d,
            next_frame_copy_d[:-1],
            keyboard_copy_d,
            keyboard_fill_copy_d,
            emb_vector_6,
            # episode_imgs[5],
        )
        all_svgs[13].scale(3.0)
        all_svgs[13].move_to([26.5, 0, 0])

        env_imgs=Group()
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_18002/real/').glob('*.png'))):
            env_imgs.add(ImageMobject(str(p)))

        for img in env_imgs:
            img.scale(2).move_to([13.4, -4.4, 0])

        self.wait()
        self.remove(everything_on_screen)
        self.play(p69_start.animate.scale(2.9).move_to([13.4, 4, 0]),
                  episode_imgs[1].animate.scale(2.9).move_to([13.4, 4, 0]),
                  episode_imgs[2].animate.scale(2.9).move_to([13.4, 4, 0]),
                  episode_imgs[3].animate.scale(2.9).move_to([13.4, 4, 0]),
                  episode_imgs[4].animate.scale(2.9).move_to([13.4, 4, 0]),
                  episode_imgs[5].animate.scale(2.9).move_to([13.4, 4, 0]),
                  all_svgs[9][-1].animate.scale(2.9).move_to([13.4, 4, 0]),
                  all_svgs[10][-1].animate.scale(2.9).move_to([13.4, 4, 0]),
                  next_frame_copy[-1].animate.scale(2.9).move_to([13.4, 4, 0]),
                  next_frame_copy_b[-1].animate.scale(2.9).move_to([13.4, 4, 0]),
                  next_frame_copy_c[-1].animate.scale(2.9).move_to([13.4, 4, 0]),
                  next_frame_copy_d[-1].animate.scale(2.9).move_to([13.4, 4, 0]),
                  run_time=8)

        self.remove(p69_start, episode_imgs[1], episode_imgs[2], episode_imgs[3], episode_imgs[4],
                    all_svgs[9][-1], all_svgs[10][-1], next_frame_copy[-1], next_frame_copy_b[-1], 
                    next_frame_copy_c[-1], next_frame_copy_d[-1])


        self.add(env_imgs[5])
        self.add(all_svgs[13][:16]) #Just first 2 panels

        step=5
        prev_step_count=Text('Step count = '+str(step), font_size=64, font='Myriad Pro')
        prev_step_count.set_color(CHILL_BROWN)
        prev_step_count.move_to([13.6, -9, 0])
        self.add(prev_step_count)

        # P72a. Loop from 5-> 18, then from 0 to 18 again. I think hard cut 
        # to P72b 
        episode_imgs_2=Group()
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_18002/real/').glob('*.png'))):
            episode_imgs_2.add(ImageMobject(str(p)))

        for img in episode_imgs_2:
            img.scale(2).move_to([13.4, 4, 0])

        # P72a: Loop 5 -> end, then 0 -> end
        max_frames = min(len(env_imgs), len(episode_imgs_2))
        print(f"env_imgs: {len(env_imgs)}, episode_imgs_2: {len(episode_imgs_2)}, using: {max_frames}")

        # Prepare remaining episode_imgs_2 that haven't been scaled/positioned yet
        # for i in range(len(episode_imgs_2)):
        #     if i == 0 or i >= 6:
        #         episode_imgs_2[i].scale(2.9).move_to([13.4, 4, 0])

        last = max_frames - 1  # final valid index
        step_sequence = list(range(6, last + 1)) + list(range(0, last + 1))

        current_env = env_imgs[5]
        current_episode = episode_imgs_2[5]
        current_step_count = prev_step_count

        self.wait(0.1)
        # self.add(episode_imgs_2[5])

        for step in step_sequence:
            new_step_count = Text('Step count = '+str(step), font_size=64, font='Myriad Pro')
            new_step_count.set_color(CHILL_BROWN)
            new_step_count.move_to([13.6, -9, 0])

            self.remove(current_env, current_episode, current_step_count)
            self.add(env_imgs[step], episode_imgs_2[step], new_step_count)
            self.remove(all_svgs[13][:16]); self.add(all_svgs[13][:16])

            current_env = env_imgs[step]
            current_episode = episode_imgs_2[step]
            current_step_count = new_step_count
            self.wait(0.5)

            




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












