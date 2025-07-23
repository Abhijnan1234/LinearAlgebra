''''
from manim import *

class Matrix3DVectorSpaceTransform(ThreeDScene):
    def construct(self):
        # Set up 3D camera orientation
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.05)  # ✅ smooth rotation

        # Axes
        axes = ThreeDAxes(x_range=[-4, 4], y_range=[-4, 4], z_range=[-2, 4])
        self.play(Create(axes))

        # Grid
        grid = self.create_grid(x_range=range(-4, 5), y_range=range(-4, 5))
        self.play(*[Create(line) for line in grid])

        # Basis vectors
        i_hat = Arrow3D(ORIGIN, [1, 0, 0], color=RED, thickness=0.02)
        j_hat = Arrow3D(ORIGIN, [0, 1, 0], color=GREEN, thickness=0.02)
        k_hat = Arrow3D(ORIGIN, [0, 0, 1], color=BLUE, thickness=0.02)
        self.play(Create(i_hat), Create(j_hat), Create(k_hat))

        # 2D labels (fixed in screen)
        matrix_tex = MathTex(
            r"A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix}"
        ).scale(0.6).to_corner(UL)
        vector_tex = MathTex(
            r"\vec{v} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}"
        ).scale(0.6).to_corner(UR)
        multiplication_tex = MathTex(
            r"A\vec{v} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix}"
            r"\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \\ 1 \end{bmatrix}"
        ).scale(0.5).to_corner(DL)

        for tex in [matrix_tex, vector_tex, multiplication_tex]:
            self.add_fixed_in_frame_mobjects(tex)
        self.play(Write(matrix_tex), Write(vector_tex), Write(multiplication_tex))

        # Original vector
        vec_start = ORIGIN
        vec_end = [1, 1, 1]
        v = Arrow3D(vec_start, vec_end, color=YELLOW, thickness=0.03)
        self.play(Create(v))

        # Label for vector (always facing screen)
        v_label = always_redraw(lambda: Tex(r"$\vec{v}$", color=YELLOW).scale(0.6).move_to(v.get_end() + 0.3 * RIGHT))
        self.add_fixed_orientation_mobjects(v_label)
        self.add(v_label)

        self.wait(1)

        # Matrix A
        A = np.array([
            [1, 0, 1],
            [0, 2, 0],
            [0, 0, 1]
        ])

        # Transform grid
        transformed_grid = [self.apply_matrix_to_line(line.copy(), A) for line in grid]

        # Transformed vector and basis
        transformed_v = Arrow3D(ORIGIN, A @ np.array([1, 1, 1]), color=ORANGE, thickness=0.03)
        transformed_i = Arrow3D(ORIGIN, A @ np.array([1, 0, 0]), color=RED, thickness=0.02)
        transformed_j = Arrow3D(ORIGIN, A @ np.array([0, 1, 0]), color=GREEN, thickness=0.02)
        transformed_k = Arrow3D(ORIGIN, A @ np.array([0, 0, 1]), color=BLUE, thickness=0.02)

        # Update label
        new_v_label = Tex(r"$A\vec{v}$", color=ORANGE).scale(0.6)
        new_v_label.move_to(transformed_v.get_end() + 0.3 * RIGHT)
        self.add_fixed_orientation_mobjects(new_v_label)

        # Animate everything
        self.play(
            *[Transform(old, new) for old, new in zip(grid, transformed_grid)],
            Transform(v, transformed_v),
            Transform(i_hat, transformed_i),
            Transform(j_hat, transformed_j),
            Transform(k_hat, transformed_k),
            FadeOut(v_label),
            FadeIn(new_v_label),
            run_time=4
        )

        self.wait(3)

    def create_grid(self, x_range, y_range):
        lines = []
        for x in x_range:
            lines.append(Line3D(start=[x, y_range[0], 0], end=[x, y_range[-1], 0], color=GREY))
        for y in y_range:
            lines.append(Line3D(start=[x_range[0], y, 0], end=[x_range[-1], y, 0], color=GREY))
        return lines

    def apply_matrix_to_line(self, line, matrix):
        start = matrix @ line.get_start()
        end = matrix @ line.get_end()
        return Line3D(start=start, end=end, color=line.color)
'''
from manim import *
import numpy as np

class MatrixProbabilityScene(ThreeDScene):
    def construct(self):
        # Set camera and rotation
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.05)

        # Step 1: Axes and cube vertices
        axes = ThreeDAxes()
        self.add(axes)

        vertices = [
            np.array([x, y, z])
            for x in [-1, 1]
            for y in [-1, 1]
            for z in [-1, 1]
        ]

        dots = [Dot3D(point, radius=0.05, color=YELLOW) for point in vertices]
        self.play(*[FadeIn(dot) for dot in dots])
        self.wait(1)

        # Step 2: Vectors from origin to vertices
        arrows = [Arrow3D(ORIGIN, v, color=WHITE) for v in vertices]
        for arrow in arrows:
            self.play(Create(arrow), run_time=0.2)

        self.wait(1)

        # Step 3: Coplanar (dependent) vectors, det = 0
        v1, v2, v3 = [np.array([1, 1, 1]), np.array([-1, -1, -1]), np.array([1, -1, 0])]
        a1 = Arrow3D(ORIGIN, v1, color=BLUE)
        a2 = Arrow3D(ORIGIN, v2, color=RED)
        a3 = Arrow3D(ORIGIN, v3, color=GREEN)

        self.play(*[FadeOut(a) for a in arrows])
        self.play(Create(a1), Create(a2), Create(a3))

        # Determinant zero
        det0_text = MathTex(
            r"\text{det}\left(\begin{bmatrix}1 & -1 & 1\\1 & -1 & -1\\1 & -1 & 0\end{bmatrix}\right) = 0"
        ).scale(0.9).to_corner(UL)
        self.add_fixed_in_frame_mobjects(det0_text)
        self.play(Write(det0_text))
        self.wait(2)

        # Step 4: Non-coplanar vectors, det ≠ 0
        self.play(FadeOut(a1), FadeOut(a2), FadeOut(a3), FadeOut(det0_text))

        u1, u2, u3 = [np.array([1, 1, 1]), np.array([-1, 1, 1]), np.array([1, -1, 1])]
        b1 = Arrow3D(ORIGIN, u1, color=BLUE)
        b2 = Arrow3D(ORIGIN, u2, color=RED)
        b3 = Arrow3D(ORIGIN, u3, color=GREEN)

        self.play(Create(b1), Create(b2), Create(b3))

        # Determinant non-ze
