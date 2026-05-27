from manimlib import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json

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



class p75b(InteractiveScene):
    def construct(self):

        num_paths_to_render=500 ## 500 for full redner
        planning_steps_to_render=30 ### 30 for full render, very slow


        svgs_to_skip=[0]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        start_im=ImageMobject(hackin_dir+'/p75b/ep2167_off25_h5/start_img.png')
        goal_im=ImageMobject(hackin_dir+'/p75b/ep2167_off25_h5/goal_img_ps.png')
        goal_im_with_border=ImageMobject(img_dir+'/goal_img_with_border.png')



        start_im.scale(1.38)
        start_im.move_to([-2.85, 0.05, 0])

        goal_im.scale(1.37)
        goal_im.move_to([3.25, 0.10, 0])

        goal_im_with_border.scale(1.37)
        goal_im_with_border.move_to([3.25, 0.10, 0])


        self.wait()
        self.play(Write(all_svgs[14]), run_time=3)

        self.wait()
        self.play(FadeIn(all_svgs[15]), FadeIn(all_svgs[16]), run_time=3)
        self.add(start_im, goal_im)
        self.add(all_svgs[15], all_svgs[16])

        # self.add(all_svgs[15]) #Start frame
        # self.add(all_svgs[16]) #Goal Frame
        # self.add(all_svgs[14]) #CEM

        d=np.load(hackin_dir+'/p75b/ep2167_off25_h5/iterations.npz')
        # for k in d.keys(): print(k, d[k].shape) 
        # paths_all (30, 500, 25, 2)                                                         
        # paths_elite (30, 30, 25, 2)                                                
        # paths_mean (30, 25, 2)
        # costs (30, 500)                     
        # topk_inds (30, 30)
        # cum_sigma (30, 25, 2)                           
        # raw_means (30, 1, 5, 10)
        # raw_vars (30, 1, 5, 10)  

        # Map 512-space (origin lower-left) to scene coords using start_im's bounds
        ll = start_im.get_corner(DL)
        ur = start_im.get_corner(UR)
        im_w = ur[0] - ll[0]
        im_h = ur[1] - ll[1]

        IMG_SIZE = 224  # not 512

        def path_to_scene(pt):
            return np.array([
                ll[0] + (pt[0] / IMG_SIZE) * im_w,
                ll[1] + ((IMG_SIZE - pt[1]) / IMG_SIZE) * im_h,   # flip y
                0.0,
            ])

        # step=0
        # path_index=45
        # path = d['paths_all'][step, path_index]
        # array([[ 30.286108, 156.19762 ],
        #        [ 35.81527 , 156.61841 ],                                                   
        #        [ 34.62877 , 154.36847 ],
        #        [ 34.440853, 153.49341 ],
        #        [ 31.434057, 155.25998 ], ...

        # --- draw one CEM path on start_im ---

        # config = json.load(open(hackin_dir + '/p75b/ep2167_off25_h5/config.json'))
        # agent_xy_img = np.array(config['agent_xy_img'])  # (2,) in 224-px space

        # step=0
        # all_path_lines=VGroup()
        # for path_index in range(500):
        #     path = d['paths_all'][step, path_index]
        #     path = d['paths_all'][step, path_index]                    # (25, 2)
        #     path_with_start = np.concatenate([agent_xy_img[None], path], axis=0)  # (26, 2)
        #     scene_pts = [path_to_scene(p) for p in path_with_start]

        #     path_line = VMobject()
        #     path_line.set_points_as_corners(scene_pts)
        #     path_line.set_stroke(MAGENTA, width=1.5)  
        #     all_path_lines.add(path_line)

        # #So we want to animate these in growing out from the center!
        # all_path_lines.set_stroke(opacity=0.2)
        # # self.add(all_path_lines)
        # for i in range(26):
        #     for p in all_path_lines:
        #         self.add()


        # import json

        config = json.load(open(hackin_dir + '/p75b/ep2167_off25_h5/config.json'))
        agent_xy_img = np.array(config['agent_xy_img'])
        step = 0

        # 1. Pre-compute the full 26-point scene-space path for every sample.
        all_full_pts = []
        for path_index in range(num_paths_to_render):
            path = d['paths_all'][step, path_index]                   # (25, 2)
            path_with_start = np.concatenate([agent_xy_img[None], path], axis=0)  # (26, 2)
            scene_pts = np.array([path_to_scene(p) for p in path_with_start])
            all_full_pts.append(scene_pts)

        # 2. Build path lines as degenerate (start->start) so they're invisible at t=0.
        all_path_lines = VGroup()
        for full_pts in all_full_pts:
            line = VMobject()
            line.set_points_as_corners([full_pts[0], full_pts[0]])
            line.set_stroke(MAGENTA, width=2.0, opacity=0.2)
            all_path_lines.add(line)

        # 3. Fade the backdrop so the cloud pops.
        self.add(all_path_lines)
        self.wait()

        # start_im.set_opacity(0.15)
        self.play(start_im.animate.set_opacity(0.1),
                  all_svgs[15].animate.set_opacity(0.1),
                  self.frame.animate.reorient(0, 0, 0, (-4.79, -1.12, 0.0), 3.10),
                  run_time=5)

        # 4. Grow one step at a time.
        n_corners  = 26          # 1 start + 25 path steps
        total_time = 4.0         # tune to taste
        dt = total_time / (n_corners - 1)
        for k in range(2, n_corners + 1):
            for line, full_pts in zip(all_path_lines, all_full_pts):
                line.set_points_as_corners(full_pts[:k])
            self.wait(dt)


        # self.wait()

        step=0
        path_index=45 #Random sample 002
        path = d['paths_all'][step, path_index]
        # path = d['paths_all'][step, path_index]                    # (25, 2)
        path_with_start = np.concatenate([agent_xy_img[None], path], axis=0)  # (26, 2)
        scene_pts = [path_to_scene(p) for p in path_with_start]

        path_line = VMobject()
        path_line.set_points_as_corners(scene_pts)
        path_line.set_stroke(YELLOW, width=4)

        path_dots = Group(*[Dot(p, radius=0.022, color=YELLOW) for p in scene_pts])
        path_dots.set_color(YELLOW)
        # self.add(path_line) #, path_dots)
        # self.add(path_dots)

        # all_path_lines.set_stroke(opacity=0.05)
        self.wait()
        self.play(all_path_lines.animate.set_stroke(opacity=0.05), 
                  self.frame.animate.reorient(0, 0, 0, (-4.35, -1.36, 0.0), 0.93),
                  FadeIn(path_line),
                  FadeIn(path_dots[0]),
                  run_time=6)

        # self.wait()
        # self.play(#self.frame.animate.reorient(0, 0, 0, (-3.66, -0.93, 0.0), 1.83),
        #           self.frame.animate.reorient(0, 0, 0, (-4.35, -1.36, 0.0), 0.93),
        #           FadeIn(path_dots[0]),
        #           run_time=5)

        self.wait()
        #Dot moves along path actually? Micky mouse style?
        self.play(ReplacementTransform(path_dots[0].copy(), path_dots[1]))
        self.play(ReplacementTransform(path_dots[1], path_dots[2]))
        self.play(ReplacementTransform(path_dots[2], path_dots[3]))
        self.play(ReplacementTransform(path_dots[3], path_dots[4]))
        self.play(ReplacementTransform(path_dots[4], path_dots[5]))

        # use this framing to draw the prediction steps on to of, 
        # then I can zoom in how ever I want. 
        # self.add(path_dots[10])
        # self.add(path_dots[15])
        # self.add(path_dots[20])
        # self.add(path_dots[25])

        rollout_ims_1=Group()
        for i in range(6):
            rollout_ims_1.add(ImageMobject(hackin_dir+'/p75b/ep2167_off25_h5/wm_rollouts/iter_000/sample_002/frame_'+str(i).zfill(2)+'.png'))

        for r in rollout_ims_1:
            r.scale(0.105)

        # self.frame.reorient(0, 0, 0, (-3.11, -0.83, 0.0), 2.35)

        # rollout_ims_1.scale(0.105)
        rollout_ims_1[0].move_to([-4.855, -0.706, 0])
        rollout_ims_1[1].move_to([-4.17, -0.706, 0])
        rollout_ims_1[2].move_to([-3.47, -0.706, 0])
        rollout_ims_1[3].move_to([-2.75, -0.706, 0])
        rollout_ims_1[4].move_to([-2.05, -0.706, 0])
        rollout_ims_1[5].move_to([-1.33, -0.706, 0])

        # self.add(rollout_ims_1)

        rollout_group=Group(*[all_svgs[i] for i in [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]])

        rollout_group[1].set_opacity(0.15)
        rollout_group[3].set_opacity(0.4)
        rollout_group[5].set_opacity(0.3)
        rollout_group[7].set_opacity(0.4)
        rollout_group[9].set_opacity(0.5)

        # self.add(rollout_group)
        rollout_group.scale(0.29)
        rollout_group.move_to([-3.1, -0.858, 0])

        self.wait()
        # frame border must stay on top throughout the fade (else it occludes then pops)
        self.play(FadeIn(rollout_group[1]),
                  self.frame.animate.reorient(0, 0, 0, (-3.55, -0.85, 0.0), 1.87),
                  FadeIn(rollout_ims_1[0]),
                  FadeIn(rollout_ims_1[1]),
                  FadeIn(rollout_group[0]),
                  run_time=5)

        self.wait()
        self.play(FadeIn(path_dots[10]),
                  FadeIn(rollout_group[3]),
                  FadeIn(rollout_ims_1[2]),
                  FadeIn(rollout_group[2]),
                  run_time=4)

        self.wait()
        self.play(FadeIn(path_dots[15]),
                  FadeIn(rollout_group[5]),
                  FadeIn(rollout_ims_1[3]),
                  FadeIn(rollout_group[4]),
                  run_time=4)

        self.wait()
        self.play(FadeIn(path_dots[20]),
                  FadeIn(rollout_group[7]),
                  FadeIn(rollout_ims_1[4]),
                  self.frame.animate.reorient(0, 0, 0, (-3.46, -0.83, 0.0), 1.95),
                  FadeIn(rollout_group[6]),
                  run_time=4)

        self.wait()
        self.play(FadeIn(path_dots[25]),
                  FadeIn(rollout_group[9]),
                  FadeIn(rollout_ims_1[5]),
                  self.frame.animate.reorient(0, 0, 0, (-3.11, -0.81, 0.0), 2.38),
                  FadeIn(rollout_group[8]),
                  run_time=4)

        #Zoom in on final positions
        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (-1.32, -0.69, 0.0), 0.69), run_time=5)



        #Zoom way out to show original goal state
        all_svgs[15].set_opacity(1.0)
        start_im.set_opacity(0.2)

        self.remove(goal_im) #The ole switcheroo
        self.remove(all_svgs[16][-1])
        self.add(goal_im_with_border)
        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (0.28, 0.25, 0.0), 7.68), run_time=6)


        goal_frame_group=Group(goal_im_with_border, all_svgs[16][:-1].copy()) #Group(all_svgs[16], goal_im)
        distance_compare_group=Group(all_svgs[27], all_svgs[28], all_svgs[29])
        distance_compare_group.scale(0.137)
        distance_compare_group.move_to([-1.03, -0.47, 0])

        self.wait()
        self.play(goal_frame_group.animate.scale(0.075).move_to([-0.75, -0.72, 0]),
                  self.frame.animate.reorient(0, 0, 0, (-0.98, -0.47, 0.0), 1.10),
                  run_time=5
            )
        # self.remove(all_svgs[16]); self.add(all_svgs[16])
        self.play(Write(distance_compare_group[0]), 
                  Write(distance_compare_group[1]), 
                  run_time=4)

        self.wait()
        self.play(Write(distance_compare_group[2]),run_time=3) 

        # goal_frame_group.scale(0.075)
        # goal_frame_group.move_to([-0.75, -0.72, 0])
        # self.add(distance_compare_group)

        # Ok now back to all the random paths, drop rollouts
        # When we get the best bath, just show rollout images
        # not full network etc 

        group_to_remove=Group(rollout_group, distance_compare_group, rollout_ims_1, goal_frame_group, 
                              path_dots[0], path_dots[5], path_dots[10], path_dots[15], path_dots[20], path_dots[25], 
                              path_line)

        # self.remove(group_to_remove)
        self.wait()
        self.play(#FadeOut(group_to_remove), 
                  # self.frame.animate.reorient(0, 0, 0, (-2.79, -1.02, 0.0), 3.58),
                  self.frame.animate.reorient(0, 0, 0, (-4.81, -1.02, 0.0), 2.66),
                  all_path_lines.animate.set_stroke(opacity=0.2),
                  start_im.animate.set_opacity(0.25),
                  all_svgs[15].animate.set_opacity(0.2),
                  run_time=6
                  )
        self.remove(group_to_remove)


        # Ok ok ok ok ok ok ok 
        # Now we need to color each path according to it's euclidean distance
        # to the embeding of the goal Img

        # --- color paths by cost at this CEM step ---
        # d['costs'] is the WM cost (Euclidean dist between predicted final emb and
        # goal emb) per candidate. Lowest cost = closest match to goal.

        # np.argmin( d['costs'][step]) #207

        # Ok i thnk we increase stroke and size
        # Bit of a problem here though, the path is going the wrong direction!
        # Gotta debub that first. 
        # Hmm unless this is the right first step - let me compare to
        # the rollut - I think i have the best rendered.... 

        costs_step = d['costs'][step, :num_paths_to_render]               # (N,)
        cmap = plt.get_cmap('viridis_r')
        norm = plt.Normalize(vmin=costs_step.min(), vmax=costs_step.max())
        path_colors_rgb = cmap(norm(costs_step))[:, :3]

        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(*(int(c * 255) for c in rgb))
        path_hex  = [rgb_to_hex(c) for c in path_colors_rgb]
        # opacities = 0.05 + 0.25 * (1.0 - norm(costs_step))                # worst→best: 0.05→0.30
        opacities = 0.2 + 0.8 * (1.0 - norm(costs_step))
        opacities = [float(o) for o in opacities]   # ← unwrap 0-d numpy scalars

        # Reorder lines + parallel arrays: worst cost first, best last (drawn on top)
        order = np.argsort(-costs_step)
        all_path_lines.submobjects = [all_path_lines.submobjects[i] for i in order]
        path_hex  = [path_hex[i]  for i in order]
        opacities = [opacities[i] for i in order]

        self.wait()
        self.play(
            *[line.animate.set_stroke(color=hc, opacity=op)
              for line, hc, op in zip(all_path_lines, path_hex, opacities)],
            run_time=5,
        )
        self.add(all_path_lines[-1])


        # Ok so this might be the right path, but I can't quite tell
        # Let me go ahead and get the dots on there

        path_index=207 #Best path for step 0
        path = d['paths_all'][step, path_index]                   # (25, 2)
        path_with_start = np.concatenate([agent_xy_img[None], path], axis=0)  # (26, 2)
        scene_pts_best_0 = [path_to_scene(p) for p in path_with_start]

        path_dots_best_0 = Group(*[Dot(p, radius=0.022, color=YELLOW) for p in scene_pts_best_0])
        path_dots_best_0.set_color(YELLOW)


        self.wait()
        self.play(all_path_lines[:-1].animate.set_stroke(opacity=0.08),
                  FadeIn(path_dots_best_0[0]),
                  FadeIn(path_dots_best_0[5]),
                  FadeIn(path_dots_best_0[10]),
                  FadeIn(path_dots_best_0[15]),
                  FadeIn(path_dots_best_0[20]),
                  FadeIn(path_dots_best_0[25]),
                  run_time=5)
        self.add(all_path_lines[-1])


        best_rollout_0=Group()
        for i in range(6):
            best_rollout_0.add(ImageMobject(hackin_dir+'/p75b/ep2167_off25_h5/wm_rollouts/iter_000/best/frame_'+str(i).zfill(2)+'.png'))

        for o in best_rollout_0:
            o.scale(0.18)

        best_rollout_0[0].move_to([-5.345, -0.31, 0])
        best_rollout_0[1].move_to([-5.57, -1.9, 0])
        best_rollout_0[2].move_to([-4.3, -1.92, 0])
        best_rollout_0[3].move_to([-5.86, -1.1, 0])
        best_rollout_0[4].move_to([-4.25, -0.25, 0])
        best_rollout_0[5].move_to([-3.8, -1.1, 0])

        all_svgs[30].scale(0.33)
        all_svgs[30].move_to([-4.83, -1.09, 0])

        self.wait()
        self.play(FadeIn(best_rollout_0), 
                  FadeIn(all_svgs[30]), 
                  self.frame.animate.reorient(0, 0, 0, (-4.75, -1.11, 0.0), 2.66), 
                  run_time=4)
        self.add(all_svgs[30])


        self.wait()
        self.remove(best_rollout_0, all_svgs[30], path_dots_best_0)
        self.play(
            *[line.animate.set_stroke(color=hc, opacity=op)
              for line, hc, op in zip(all_path_lines[-30:], path_hex[-30:], opacities[-30:])],
            run_time=5,
        )
        self.add(all_path_lines[-1])



        self.wait()
        self.remove(all_path_lines)

        # Ok now draw in all paths in step from their starting points
        # Start with them already colored by cost
        # Then drop the opacity on all but top 30. 

        # === Step 1: CEM iteration 1 paths ===
        step = 1

        # 1. Pre-compute scene-space paths
        all_full_pts_1 = []
        for path_index in range(num_paths_to_render):
            path = d['paths_all'][step, path_index]
            path_with_start = np.concatenate([agent_xy_img[None], path], axis=0)
            scene_pts = np.array([path_to_scene(p) for p in path_with_start])
            all_full_pts_1.append(scene_pts)

        # 2. Costs → colors + opacities (re-normalize per step so contrast doesn't collapse)
        costs_step_1 = d['costs'][step, :num_paths_to_render]
        norm_1 = plt.Normalize(vmin=costs_step_1.min()*1.4, vmax=costs_step_1.max()) #SWAG adjustement here
        path_hex_1  = [rgb_to_hex(c) for c in cmap(norm_1(costs_step_1))[:, :3]]
        opacities_1 = [float(o) for o in (0.1 + 0.7 * (1.0 - norm_1(costs_step_1)))]

        # 3. Sort worst→best so best draws on top
        order_1       = np.argsort(-costs_step_1)
        all_full_pts_1 = [all_full_pts_1[i] for i in order_1]
        path_hex_1     = [path_hex_1[i]     for i in order_1]
        opacities_1    = [opacities_1[i]    for i in order_1]

        # 4. Build lines degenerate but already colored
        all_path_lines_1 = VGroup()
        for full_pts, hc, op in zip(all_full_pts_1, path_hex_1, opacities_1):
            line = VMobject()
            line.set_points_as_corners([full_pts[0], full_pts[0]])
            line.set_stroke(color=hc, width=2.0, opacity=op)
            all_path_lines_1.add(line)

        self.add(all_path_lines_1)

        # 5. Grow corner-by-corner
        n_corners  = 26
        total_time = 4.0
        dt = total_time / (n_corners - 1)
        for k in range(2, n_corners + 1):
            for line, full_pts in zip(all_path_lines_1, all_full_pts_1):
                line.set_points_as_corners(full_pts[:k])
            self.wait(dt)

        # 6. Fade everything except top-30 (sorted worst-first, so top-30 = last 30)
        self.wait()
        self.play(
            *[line.animate.set_stroke(opacity=0.05)
              for line in all_path_lines_1[:-30]],
            run_time=3,
        )
        self.add(all_path_lines_1[-30:])  # keep elites on top after the fade

        # === Steps 2..29: same pattern, automated ===

        # Pacing knobs (tune these to taste; lower = snappier)
        GROW_TIME       = 1.2   # how long the corner-by-corner reveal takes
        FADE_TIME       = 0.8   # fade non-elites to bg
        INTER_STEP_WAIT = 0.2   # breath between iterations

        # Hold a reference to the previous step's group so we can swap cleanly
        prev_lines = all_path_lines_1

        for step in range(2, planning_steps_to_render):
            # 1. Scene-space paths
            all_full_pts_k = []
            for path_index in range(num_paths_to_render):
                path = d['paths_all'][step, path_index]
                path_with_start = np.concatenate([agent_xy_img[None], path], axis=0)
                scene_pts = np.array([path_to_scene(p) for p in path_with_start])
                all_full_pts_k.append(scene_pts)

            # 2. Per-step colors + opacities
            costs_step_k = d['costs'][step, :num_paths_to_render]
            norm_k = plt.Normalize(vmin=costs_step_k.min()*1.5, vmax=costs_step_k.max()) #Keeping my swag adjustement
            path_hex_k  = [rgb_to_hex(c) for c in cmap(norm_k(costs_step_k))[:, :3]]
            opacities_k = [float(o) for o in (0.2 + 0.8 * (1.0 - norm_k(costs_step_k)))]

            # 3. Sort worst→best
            order_k        = np.argsort(-costs_step_k)
            all_full_pts_k = [all_full_pts_k[i] for i in order_k]
            path_hex_k     = [path_hex_k[i]     for i in order_k]
            opacities_k    = [opacities_k[i]    for i in order_k]

            # 4. Build degenerate, already-colored lines
            all_path_lines_k = VGroup()
            for full_pts, hc, op in zip(all_full_pts_k, path_hex_k, opacities_k):
                line = VMobject()
                line.set_points_as_corners([full_pts[0], full_pts[0]])
                line.set_stroke(color=hc, width=2.0, opacity=op)
                all_path_lines_k.add(line)

            # 5. Swap: remove previous step, add this step
            self.remove(prev_lines)
            self.add(all_path_lines_k)

            # 6. Grow corner-by-corner
            n_corners = 26
            dt = GROW_TIME / (n_corners - 1)
            for kk in range(2, n_corners + 1):
                for line, full_pts in zip(all_path_lines_k, all_full_pts_k):
                    line.set_points_as_corners(full_pts[:kk])
                self.wait(dt)

            # 7. Fade non-elites, elites on top
            self.play(
                *[line.animate.set_stroke(opacity=0.05)
                  for line in all_path_lines_k[:-30]],
                run_time=FADE_TIME,
            )
            self.add(all_path_lines_k[-30:])

            self.wait(INTER_STEP_WAIT)

            prev_lines = all_path_lines_k


        # === Final executed path on the real environment ===
        rollout = np.load(hackin_dir + '/p75b/ep2167_off25_h5_rollout/rollout.npz')

        SCALE_512_TO_224 = IMG_SIZE / 512.0   # 224/512 = 0.4375
        agent_path_224 = rollout['states'][:, :2] * SCALE_512_TO_224   # (26, 2)
        final_scene_pts = [path_to_scene(p) for p in agent_path_224]

        final_path_line = VMobject()
        final_path_line.set_points_as_corners(final_scene_pts)
        final_path_line.set_stroke(YELLOW, width=4)

        final_path_dots = Group(*[Dot(p, radius=0.022, color=YELLOW) for p in final_scene_pts])
        final_path_dots.set_color(YELLOW)

        # Swap: planned cloud out, final plan in
        self.wait()
        self.play(
            FadeOut(prev_lines),
            FadeIn(final_path_line),
            FadeIn(final_path_dots),
            run_time=3,
        )



        start_im.set_opacity(1.0)
        all_svgs[15].set_opacity(1.0)
        self.add(goal_im, all_svgs[16])
        # goal_im.set_opacity(1.0)
        # all_svgs[16].set_opacity(1.0)


        final_imgs=Group()
        for i in range(26):
            final_imgs.add(ImageMobject(hackin_dir+'/p75b/ep2167_off25_h5_rollout/frame_'+str(i).zfill(3)+'.png'))

        for im in final_imgs:
            im.scale(1.38)
            im.move_to([-2.85, 0.05, 0])


        self.remove(all_svgs[15][:-1]) #Remove start frame label bfore we zoom out
        self.wait()
        self.play(self.frame.animate.reorient(0, 0, 0, (0.2, 0.22, 0.0), 7.64), run_time=8)

        self.wait()
        self.remove(start_im)
        for i in range(len(final_imgs)-1):
            if i>0: self.remove(final_imgs[i])
            self.add(final_imgs[i+1])
            self.add(all_svgs[15])
            self.add(final_path_line)
            self.add(final_path_dots)
            self.wait(0.2)


        #Holy shit that was a crazy scene. 



        


        self.wait(20)
        self.embed()





class p71b(InteractiveScene):
    def construct(self):

        # Ok yeah I think i want to run some other variants, but this is good enough for 
        # now going to be. Also annoying tha the final column isnt' finishing, should address that. 

        episodes=[363, 2319, 7965, 10722] #Might want to try some variants


        svgs_to_skip=[0]
        svg_files=list(sorted(svg_dir.glob('*.svg')))
        all_svgs=Group()
        for i, svg_file in enumerate(svg_files): 
            if i in svgs_to_skip: continue
            svg_image=SVGMobject(str(svg_file))
            svg_image.scale(4.0)
            all_svgs.add(svg_image[1:])

        imgs_1=[Group(), Group()]
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_'+str(episodes[0]).zfill(4)+'/wm/').glob('*.png'))):
            imgs_1[0].add(ImageMobject(str(p)))
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_'+str(episodes[0]).zfill(4)+'/real/').glob('*.png'))):
            imgs_1[1].add(ImageMobject(str(p)))

        imgs_2=[Group(), Group()]
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_'+str(episodes[1]).zfill(4)+'/wm/').glob('*.png'))):
            imgs_2[0].add(ImageMobject(str(p)))
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_'+str(episodes[1]).zfill(4)+'/real/').glob('*.png'))):
            imgs_2[1].add(ImageMobject(str(p)))

        imgs_3=[Group(), Group()]
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_'+str(episodes[2]).zfill(4)+'/wm/').glob('*.png'))):
            imgs_3[0].add(ImageMobject(str(p)))
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_'+str(episodes[2]).zfill(4)+'/real/').glob('*.png'))):
            imgs_3[1].add(ImageMobject(str(p)))

        imgs_4=[Group(), Group()]
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_'+str(episodes[3]).zfill(4)+'/wm/').glob('*.png'))):
            imgs_4[0].add(ImageMobject(str(p)))
        for i, p in enumerate(sorted(Path(hackin_dir+'/p71/ep_'+str(episodes[3]).zfill(4)+'/real/').glob('*.png'))):
            imgs_4[1].add(ImageMobject(str(p)))

        print(len(imgs_1[0]), len(imgs_2[0]), len(imgs_3[0]), len(imgs_4[0]))
        print(len(imgs_1[1]), len(imgs_2[1]), len(imgs_3[1]), len(imgs_4[1]))

        for img in imgs_1[0]:
            img.scale(0.67).move_to([-4.65, 1.38, 0])

        for img in imgs_1[1]:
            img.scale(0.67).move_to([-4.65, -1.41, 0])

        for img in imgs_2[0]:
            img.scale(0.67).move_to([-1.59, 1.38, 0])

        for img in imgs_2[1]:
            img.scale(0.67).move_to([-1.59, -1.41, 0])

        for img in imgs_3[0]:
            img.scale(0.67).move_to([-1.59+2.96, 1.39, 0])

        for img in imgs_3[1]:
            img.scale(0.67).move_to([-1.59+2.99, -1.41, 0])

        for img in imgs_4[0]:
            img.scale(0.67).move_to([-1.59+2*3.01, 1.39, 0])

        for img in imgs_4[1]:
            img.scale(0.67).move_to([-1.59+2*3.03, -1.41, 0])
        
        # imgs_1[0][0].scale(0.67)
        # imgs_1[0][0].move_to([-4.65, 1.38, 0])
        # self.add(imgs_1[0][0])

        # imgs_1[1][0].scale(0.67)
        # imgs_1[1][0].move_to([-4.65, -1.41, 0])
        # self.add(imgs_1[1][0])

        # imgs_2[0][0].scale(0.67)
        # imgs_2[0][0].move_to([-1.59, 1.38, 0])
        # self.add(imgs_2[0][0])

        # imgs_2[1][0].scale(0.67)
        # imgs_2[1][0].move_to([-1.59, -1.41, 0])
        # self.add(imgs_2[1][0])

        # imgs_3[0][0].scale(0.67)
        # imgs_3[0][0].move_to([-1.59+2.96, 1.39, 0])
        # self.add(imgs_3[0][0])

        # imgs_3[1][0].scale(0.67)
        # imgs_3[1][0].move_to([-1.59+2.99, -1.41, 0])
        # self.add(imgs_3[1][0])

        # imgs_4[0][0].scale(0.67)
        # imgs_4[0][0].move_to([-1.59+2*3.01, 1.39, 0])
        # self.add(imgs_4[0][0])

        # imgs_4[1][0].scale(0.67)
        # imgs_4[1][0].move_to([-1.59+2*3.03, -1.41, 0])
        # self.add(imgs_4[1][0])

        count=0
        self.add(imgs_1[0][count], imgs_1[1][count],
                imgs_2[0][count], imgs_2[1][count],
                imgs_3[0][count], imgs_3[1][count],
                imgs_4[0][count], imgs_4[1][count])
        self.add(all_svgs[13])

        prev_step_count_1=Text('Step count = '+str(count), font_size=24, font='Myriad Pro')
        prev_step_count_1.set_color(CHILL_BROWN)
        prev_step_count_1.move_to([-4.6, -3, 0])
        self.add(prev_step_count_1)

#       Step counts beneath the other three pairs
        prev_step_count_2 = Text('Step count = '+str(count), font_size=24, font='Myriad Pro')
        prev_step_count_2.set_color(CHILL_BROWN)
        prev_step_count_2.move_to([-1.59, -3, 0])
        self.add(prev_step_count_2)

        prev_step_count_3 = Text('Step count = '+str(count), font_size=24, font='Myriad Pro')
        prev_step_count_3.set_color(CHILL_BROWN)
        prev_step_count_3.move_to([-1.59+3.0, -3, 0])
        self.add(prev_step_count_3)

        prev_step_count_4 = Text('Step count = '+str(count), font_size=24, font='Myriad Pro')
        prev_step_count_4.set_color(CHILL_BROWN)
        prev_step_count_4.move_to([-1.59+2*3.01, -3, 0])
        self.add(prev_step_count_4)

        self.wait(0.3)

        # Loop through frames, holding last frame for sets that end early
        all_imgs = [imgs_1, imgs_2, imgs_3, imgs_4]
        prev_step_counts = [prev_step_count_1, prev_step_count_2,
                            prev_step_count_3, prev_step_count_4]
        x_positions = [-4.6, -1.59, -1.59+2.96, -1.59+2*3.01]
        max_length = max(len(imgs[0]) for imgs in all_imgs)

        for step in range(1, max_length):
            # Lift the svg overlay so swapped images don't sit on top of it
            self.remove(all_svgs[13])

            for i, imgs in enumerate(all_imgs):
                prev_idx = min(step - 1, len(imgs[0]) - 1)
                curr_idx = min(step, len(imgs[0]) - 1)

                # Only swap if this set hasn't ended yet
                if prev_idx != curr_idx:
                    self.remove(imgs[0][prev_idx], imgs[1][prev_idx])
                    self.add(imgs[0][curr_idx], imgs[1][curr_idx])

                    new_sc = Text('Step count = '+str(curr_idx),
                                  font_size=24, font='Myriad Pro')
                    new_sc.set_color(CHILL_BROWN)
                    new_sc.move_to([x_positions[i], -3, 0])
                    self.remove(prev_step_counts[i])
                    self.add(new_sc)
                    prev_step_counts[i] = new_sc

            # Put the frame svg back on top
            self.add(all_svgs[13])
            self.wait(0.3)

        self.wait()


        self.wait(20)
        self.embed()



class p69_71a(InteractiveScene):
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












