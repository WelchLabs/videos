import manimgx as gx
from _2026.vla.trajectory_storyboard import DIFFUSION_DIR
from welch_scene import WelchScene, image_at_height


CHILL_BROWN = '#948979'


class P50(WelchScene):
    def construct(self):

        img_height = 4.0

        all_images = []
        for idx in range(100):
            path = DIFFUSION_DIR / f"step_{idx:03d}.png"
            img = image_at_height(path)
            img.scale_to_fit_height(img_height)
            all_images.append(img)

        self.add(all_images[0])
        self.wait()

        label = gx.Text("REALISTIC_VISION_V5.1", font_size=24).set_color(CHILL_BROWN)
        label.move_to(all_images[0].get_center() + gx.DOWN * (img_height / 2 + 0.3))
        label.align_to(all_images[0], gx.RIGHT)

        self.add(label)

        index_label = gx.Text("1", font_size=24).set_color(CHILL_BROWN)
        index_label.move_to(all_images[0].get_center() + gx.DOWN * (img_height / 2 + 0.3))
        index_label.align_to(all_images[0], gx.LEFT)

        self.add(index_label)

        for i in range(1, 100):
            self.remove(all_images[i - 1])
            self.add(all_images[i])
            new_index_label = gx.Text(str(i + 1), font_size=24).set_color(CHILL_BROWN)
            new_index_label.move_to(all_images[0].get_center() + gx.DOWN * (img_height / 2 + 0.3))
            new_index_label.align_to(all_images[0], gx.LEFT)
            self.remove(index_label)
            index_label = new_index_label
            self.add(index_label)
            self.wait(1/15)

        self.wait()

        expand_indices = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        expand_images = []
        for idx in expand_indices:
            path = DIFFUSION_DIR / f"step_{idx:03d}.png"
            img = image_at_height(path)
            img.scale_to_fit_height(img_height)
            img.move_to(all_images[99].get_center())
            expand_images.append(img)

        # Target arrangement
        target_height = 1.8
        buff = 0.1

        targets = gx.Group()
        for img in expand_images:
            t = img.copy()
            t.scale_to_fit_height(target_height)
            targets.add(t)
        targets.arrange(gx.RIGHT, buff=buff)
        targets.move_to(gx.ORIGIN)

        step_labels = []
        start_label_offset = gx.DOWN * (img_height / 2 + 0.3)
        target_label_offset = gx.DOWN * (target_height / 2 + 0.3 * (target_height / img_height))
        for i, idx in enumerate(expand_indices):
            sl = gx.Text(str(idx + 1), font_size=16).set_color(CHILL_BROWN)
            sl.move_to(all_images[99].get_center() + start_label_offset)
            sl.align_to(all_images[99], gx.LEFT)
            step_labels.append(sl)

        target_label_positions = []
        for i, idx in enumerate(expand_indices):
            tl = gx.Text(str(idx + 1), font_size=16).set_color(CHILL_BROWN)
            tl.move_to(targets[i].get_center() + target_label_offset)
            tl.align_to(targets[i], gx.LEFT)
            target_label_positions.append(tl)

        self.remove(all_images[99], label, index_label)
        for img in expand_images:
            self.add(img)
        for sl in step_labels:
            self.add(sl)

        strip_width = targets.width + 1.0
        strip_height = targets.height + 1.5
        frame = self.camera.frame
        target_frame_width = max(strip_width, strip_height * frame.width / frame.height)

        self.play(
            *[
                img.animate.match_height(targets[i]).move_to(targets[i].get_center())
                for i, img in enumerate(expand_images)
            ],
            *[
                sl.animate.move_to(target_label_positions[i].get_center())
                for i, sl in enumerate(step_labels)
            ],
            frame.animate.scale_to_fit_width(target_frame_width).move_to(targets.get_center()),
            run_time=3,
        )
        self.wait()
