from manim import *
import numpy as np


np.random.seed(42)

def get_spectrum_data(
    noise_level: float
) -> np.ndarray:
    x_values = np.arange(1, 51, 1)
    y_values = np.random.normal(loc=0, scale=noise_level, size=x_values.shape)
    y_values += 10 * np.exp(-0.5 * (x_values - 15)**2 / 2**2 )
    y_values += 3 * np.exp(-0.5 * (x_values - 20)**2 / 1**2 )
    y_values += 8 * np.exp(-0.5 * (x_values - 40)**2 / 2.5**2 )
    return np.column_stack((x_values, y_values))


class CNNExample(ZoomedScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._section_i = 1

    def section(self):
        if self._section_i > 1:
            self.wait(0.1)
        self.next_section(f"{self._section_i}-CNNExample")
        self._section_i += 1

    def construct(self):
        self.section()
        spectrum_data = get_spectrum_data(noise_level=0.5)
        grid = Axes(
            x_range=[1, 55],
            y_range=[-2, 12],
            x_axis_config={
                "numbers_to_include": [0, 10, 20, 30, 40, 50],
            },
            y_axis_config={
                "numbers_to_include": [0, 5, 10],
            },
            tips=False,
        )
        spectrum = grid.plot_line_graph(
            spectrum_data[:,0],
            spectrum_data[:,1],
            add_vertex_dots=False,
            line_color=RED,
        )
        self.play(Create(grid), run_time=1.5, rate_func=linear)
        self.play(Create(spectrum), run_time=2, rate_func=smooth)
        self.add(grid, spectrum)
        self.section()
        self.play(
            grid.animate.rotate(-PI/2, axis=RIGHT, about_point=spectrum.get_center()),
            spectrum.animate.rotate(-PI/2, axis=RIGHT, about_point=spectrum.get_center()
                                    ).shift(UP * 0.36 + RIGHT * 0.7).scale(1.15),
            run_time=2,
        )
        self.play(
            FadeOut(grid),
            run_time=0.5,
        )
        boxes = VGroup()
        for x, y in spectrum_data:
            box = Square(side_length=0.25, fill_opacity=0.2, fill_color=RED, stroke_color=RED, stroke_width=1)
            box.move_to((x/4 - 25/4, 0, 0))
            text = Text(str(round(y, 1)), font_size=8).move_to(box.get_center())
            box.add(text)
            boxes.add(box)
        text = Text(str(round(spectrum_data[1,-1], 1)), font_size=8).move_to(boxes[-1].get_center())
        boxes[-1].add(text)

        self.play(
            FadeOut(spectrum),
            FadeIn(boxes),
            run_time=1,
        )
        self.section()
        zoom_group = VGroup(*boxes[:10])

        self.play(
            self.camera.frame.animate.move_to(zoom_group.get_center()).scale(0.2),
            run_time=1
        )

        self.section()

        kernel = ["$a_1$", "$a_2$", "$a_3$", "$a_4$", "$a_5$"]
        kernel_boxes = VGroup()
        for i, k in enumerate(kernel):
            box = Square(side_length=0.25, fill_opacity=0.2, fill_color=BLUE, stroke_color=BLUE, stroke_width=2)
            box.move_to(boxes[i].get_center() + UP * 0.5)
            text = Tex(k, font_size=12, color=WHITE).move_to(box.get_center())
            box.add(text)
            kernel_boxes.add(box)

        # Animate kernel sliding over the input
        result_boxes = VGroup()
        for i in range(12):
            self.play(kernel_boxes.animate.move_to(VGroup(*boxes[i:i+5]).get_center() + UP * 0.5), run_time=0.3)
            highlights = VGroup(*[boxes[j].copy().set_stroke(YELLOW, 2) for j in range(i, i+5)])
            result_box = Square(side_length=0.25, fill_opacity=0.2, fill_color=BLUE, stroke_color=BLUE, stroke_width=2)
            result_box.move_to(VGroup(*boxes[i:i+5]).get_center() + DOWN * 0.5)
            if i == 0:
                links = VGroup()
                mult_signs = VGroup()
                for k_box, s_box in zip(kernel_boxes, boxes[i:i+5]):
                    line = Line(k_box.get_bottom(), s_box.get_top(), stroke_width=2, color=WHITE)
                    mult = Tex(r"$\times$", font_size=18).move_to(line.get_center() + 0.13 * UP)
                    links.add(line)
                    mult_signs.add(mult)
                self.play(Create(links), FadeIn(mult_signs), run_time=0.5)

                sum_links = VGroup()
                for s_box in boxes[i:i+5]:
                    line = Line(s_box.get_bottom(), result_box.get_top(), stroke_width=2, color=WHITE)
                    sum_links.add(line)
                sum_tex = Tex(r"$\sum$", font_size=18).move_to(result_box.get_center())
                self.play(FadeIn(highlights), run_time=0.1)
                self.play(FadeIn(result_box), run_time=0.2)
                self.play(Create(sum_links), FadeIn(sum_tex), run_time=0.4)

                self.section()
                self.play(FadeOut(links), FadeOut(mult_signs), FadeOut(sum_links), FadeOut(sum_tex), run_time=0.3)
            else:
                self.play(FadeIn(highlights), run_time=0.1)
                self.play(FadeIn(result_box), run_time=0.2)
            self.play(FadeOut(highlights), run_time=0.1)
            result_boxes.add(result_box)

        self.section()

        self.play(result_boxes.animate.move_to(result_boxes.get_center() + DOWN * 0.3), run_time=0.5)

        new_kernel = ["$b_1$", "$b_2$", "$b_3$", "$b_4$", "$b_5$"]
        new_kernel_boxes = VGroup()
        for i, k in enumerate(new_kernel):
            box = Square(side_length=0.25, fill_opacity=0.2, fill_color=GREEN, stroke_color=GREEN, stroke_width=2)
            box.move_to(boxes[i].get_center() + UP * 0.5)
            text = Tex(k, font_size=12, color=WHITE).move_to(box.get_center())
            box.add(text)
            new_kernel_boxes.add(box)

        new_result_boxes = VGroup()
        for i in range(12):
            self.play(new_kernel_boxes.animate.move_to(VGroup(*boxes[i:i+5]).get_center() + UP * 0.5), run_time=0.3)
            highlights = VGroup(*[boxes[j].copy().set_stroke(YELLOW, 2) for j in range(i, i+5)])
            self.play(FadeIn(highlights), run_time=0.1*1)
            result_box = Square(side_length=0.25, fill_opacity=0.2, fill_color=GREEN, stroke_color=GREEN,
                                stroke_width=2)
            result_box.move_to(VGroup(*boxes[i:i+5]).get_center() + DOWN * 0.5)
            self.play(FadeIn(result_box), run_time=0.2*1)
            self.play(FadeOut(highlights), run_time=0.1*1)
            new_result_boxes.add(result_box)

        self.section()
        vdots = Tex(r"$\vdots$", font_size=24).move_to(new_result_boxes.get_center() + LEFT * 0.7 + UP * 0.1)
        self.play(
            result_boxes.animate.move_to(result_boxes.get_center() + DOWN * 0.3),
            new_result_boxes.animate.move_to(new_result_boxes.get_center() + DOWN * 0.3),
            GrowFromEdge(vdots, UP),
        )
        self.remove(new_kernel_boxes, kernel_boxes)

        self.section()

        for i in range(46-12):
            result_boxes.add(Square(side_length=0.25, fill_opacity=0.2, fill_color=BLUE, stroke_color=BLUE,
                                    stroke_width=2).move_to(result_boxes[-1].get_center() + RIGHT * 0.25))
            new_result_boxes.add(Square(side_length=0.25, fill_opacity=0.2, fill_color=GREEN, stroke_color=GREEN,
                                        stroke_width=2).move_to(new_result_boxes[-1].get_center() + RIGHT * 0.25))

        self.play(
            self.camera.frame.animate.move_to(ORIGIN).scale(5),
            ShrinkToCenter(vdots),
        )
        self.play(
            boxes.animate.move_to(LEFT * 15),
            result_boxes.animate.move_to(new_result_boxes.get_center() + UP * 3 + RIGHT * 11.50),
            new_result_boxes.animate.move_to(new_result_boxes.get_center() + UP * 3),
        )
        self.remove(vdots)
        self.remove(boxes)

        self.section()

        output_labels = [
            "$A_1$", r"$\mu_1$", r"$\sigma_1$", "$A_2$", r"$\mu_2$", r"$\sigma_2$", "$A_3$", r"$\mu_3$", r"$\sigma_3$"
        ]
        output_boxes = VGroup()
        for i, label in enumerate(output_labels):
            box = Square(side_length=0.8, fill_opacity=0.3, fill_color=PURPLE, stroke_color=PURPLE, stroke_width=2)
            box.move_to(ORIGIN + DOWN + RIGHT * (i - 4) * 1)
            text = Tex(label, font_size=50, color=WHITE).move_to(box.get_center())
            box.add(text)
            output_boxes.add(box)
        self.play(FadeIn(output_boxes), run_time=1)

        self.section()

        # Draw lines from all green and blue boxes to each output box (dense connections)
        lines = VGroup()
        for box_group in [new_result_boxes, result_boxes[:15]]:
            for fc_box in box_group:
                for out_box in output_boxes:
                    line = Line(fc_box.get_bottom(), out_box.get_top(), stroke_width=1, color=GREY_B)
                    lines.add(line)
        self.play(Create(lines), run_time=5)

        self.section()

        grid.rotate(PI/2, axis=RIGHT, about_point=spectrum.get_center()).move_to(
            self.camera.frame.get_center() + DOWN * 1).scale(0.9)

        self.play(
            new_result_boxes.animate.move_to(new_result_boxes.get_center() + UP * 2.5),
            result_boxes.animate.move_to(result_boxes.get_center() + UP * 2.5),
            FadeOut(lines, shift=UP, scale=1.2),
            output_boxes.animate.move_to(output_boxes.get_center() + UP * 3),
            FadeIn(new_spectrum := grid.plot_line_graph(
                spectrum_data[:,0],
                spectrum_data[:,1],
                add_vertex_dots=False,
                line_color=RED,
                stroke_width=4,
            )),
        )

        gaussian_params = [
            {"A": 10, "mu": 15, "sigma": 2, "color": GOLD_A},
            {"A": 3, "mu": 20, "sigma": 1, "color": PURE_GREEN},
            {"A": 8, "mu": 40, "sigma": 2.5, "color": BLUE_C},
        ]
        gaussian_indices = [
            [0, 1, 2],  # A1, mu1, sigma1
            [3, 4, 5],  # A2, mu2, sigma2
            [6, 7, 8],  # A3, mu3, sigma3
        ]

        x = np.linspace(1, 50, 200)
        all_highlights = VGroup()
        for i, params in enumerate(gaussian_params):
            # Draw the Gaussian curve
            y = params["A"] * np.exp(-0.5 * (x - params["mu"])**2 / params["sigma"]**2)
            gauss_curve = grid.plot_line_graph(
                x, y, add_vertex_dots=False, line_color=params["color"], stroke_width=6
            )
            # Highlight corresponding output boxes
            highlights = VGroup()
            for idx in gaussian_indices[i]:
                highlight_box = output_boxes[idx].copy().set_stroke(params["color"], 2).set_fill(opacity=0.6)
                highlights.add(highlight_box)
                all_highlights.add(highlight_box)
            self.play(
                Create(gauss_curve),
                FadeIn(highlights),
                run_time=1.2
            )

        self.play(
            FadeOut(new_spectrum),
        )

        self.section()

        output_boxes_animation = [
            [
                box_type[0].animate.shift(DOWN * 2),
                box_type[1].animate.move_to(box_type[0].get_center() + DOWN * 2 + DOWN * 1),
                box_type[2].animate.move_to(box_type[0].get_center() + DOWN * 2 + DOWN * 2),
                box_type[3].animate.shift(RIGHT * 0.5 + DOWN * 2),
                box_type[4].animate.move_to(box_type[3].get_center() + RIGHT * 0.5 + DOWN * 2 + DOWN * 1),
                box_type[5].animate.move_to(box_type[3].get_center() + RIGHT * 0.5 + DOWN * 2 + DOWN * 2),
                box_type[6].animate.shift(DOWN * 2 + RIGHT * 2.5),
                box_type[7].animate.move_to(box_type[6].get_center() + DOWN * 2 + RIGHT * 2.5 + DOWN * 1),
                box_type[8].animate.move_to(box_type[6].get_center() + DOWN * 2 + RIGHT * 2.5 + DOWN * 2),
            ]
            for box_type in [output_boxes, all_highlights]
        ]

        self.play(
            *output_boxes_animation[0],
            *output_boxes_animation[1],
        )
