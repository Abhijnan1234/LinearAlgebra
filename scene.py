from manim import *

class Determinant3DProperlyAnchored(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.05)

        axes = ThreeDAxes()
        self.add(axes)

        # Transformation matrix
        A = np.array([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])
        det_val = round(np.linalg.det(A))

        # Original unit cube from origin
        unit_cube = self.create_cube_from_origin(np.identity(3), color=WHITE, opacity=0.3)
        self.add(unit_cube)

        # Basis vectors
        origin = np.array([0, 0, 0])
        i_arrow = Arrow3D(origin, [1, 0, 0], color=BLUE)
        j_arrow = Arrow3D(origin, [0, 1, 0], color=RED)
        k_arrow = Arrow3D(origin, [0, 0, 1], color=GREEN)

        self.play(Create(i_arrow), Create(j_arrow), Create(k_arrow))

        # Transform basis
        Ai = A @ np.array([1, 0, 0])
        Aj = A @ np.array([0, 1, 0])
        Ak = A @ np.array([0, 0, 1])

        Ai_arrow = Arrow3D(origin, Ai, color=BLUE)
        Aj_arrow = Arrow3D(origin, Aj, color=RED)
        Ak_arrow = Arrow3D(origin, Ak, color=GREEN)

        self.play(
            Transform(i_arrow, Ai_arrow),
            Transform(j_arrow, Aj_arrow),
            Transform(k_arrow, Ak_arrow),
        )

        # Animate unit cube becoming parallelepiped
        transformed_cube = self.create_cube_from_origin(A, color=TEAL, opacity=0.4)
        self.play(Transform(unit_cube, transformed_cube), run_time=2)

        # Determinant and matrix display (fixed to 2D)
        det_text = MathTex(r"\text{det}(A) = " + str(det_val)).scale(1.2)
        matrix_tex = MathTex(r"A = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}").scale(1)

        det_text.to_corner(UL)
        matrix_tex.next_to(det_text, DOWN, aligned_edge=LEFT)

        self.add_fixed_in_frame_mobjects(det_text, matrix_tex)
        self.play(Write(det_text), Write(matrix_tex))
        self.wait(3)

    def create_cube_from_origin(self, matrix, color=WHITE, opacity=0.3):
        corners = [np.array([x, y, z]) for x in [0, 1] for y in [0, 1] for z in [0, 1]]
        transformed = [matrix @ c for c in corners]

        def face(indices):
            return Polygon3D(*[transformed[i] for i in indices],
                             color=color, fill_opacity=opacity, stroke_color=WHITE)

        faces = [
            face([0, 1, 3, 2]),  # Bottom
            face([4, 5, 7, 6]),  # Top
            face([0, 1, 5, 4]),  # Front
            face([2, 3, 7, 6]),  # Back
            face([0, 2, 6, 4]),  # Left
            face([1, 3, 7, 5])   # Right
        ]
        return VGroup(*faces)

# Custom class for 3D polygons
class Polygon3D(Polygon):
    def __init__(self, *points, **kwargs):
        super().__init__(*points, **kwargs)
        self.make_3d()

    def make_3d(self):
        for submob in self.family_members_with_points():
            submob.set_shade_in_3d(True)

'''
from manim import *

class OrthogonalTransform(LinearTransformationScene):
    def __init__(self):
        super().__init__(
            show_coordinates=True,
            leave_ghost_vectors=True,
            include_foreground_plane=True,
            foreground_plane_kwargs={"x_range": [-6, 6], "y_range": [-6, 6]},
        )

    def construct(self):
        # Orthogonal (rotation) matrix
        matrix = [[0, -1], [1, 0]]
        matrix_tex = MathTex(
            r"A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}"
        ).to_corner(UL)
        self.add(matrix_tex)

        # Original vector
        vector = Vector([2, 1], color=BLUE)
        vector_label = MathTex(r"\vec{v} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}")
        vector_label.next_to(vector.get_end(), RIGHT + UP * 0.2)
        self.add_vector(vector)
        self.add(vector_label)

        # Matrix multiplication on screen
        multiplication = MathTex(
            r"A \vec{v} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}"
            r"\begin{bmatrix} 2 \\ 1 \end{bmatrix}"
            r"= \begin{bmatrix} -1 \\ 2 \end{bmatrix}"
        ).scale(0.7).to_corner(DL)
        self.play(Write(multiplication))
        self.wait(1)

        # Apply transformation
        self.apply_matrix(matrix)
        self.wait(0.5)

        # Change vector label after rotation
        new_label = MathTex(r"A\vec{v} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}")
        new_label.next_to(vector.get_end(), RIGHT + UP * 0.2)
        self.play(Transform(vector_label, new_label))

        self.wait(2)


from manim import *

class CleanIntegerMatrixTransform(LinearTransformationScene):
    def __init__(self):
        super().__init__(
            show_coordinates=True,
            leave_ghost_vectors=True,
            include_foreground_plane=True,
            foreground_plane_kwargs={"x_range": [-6, 6], "y_range": [-6, 6]},
        )

    def construct(self):
        # Matrix A = [[1, 1], [0, 1]]
        matrix = [[1, 1], [0, 1]]
        matrix_tex = MathTex(
            r"A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}"
        ).to_corner(UL)
        self.add(matrix_tex)

        # Original vector v
        vector = Vector([2, 1], color=BLUE)
        vector_label = MathTex(r"\vec{v} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}")
        vector_label.next_to(vector.get_end(), RIGHT + UP * 0.2)
        self.add_vector(vector)
        self.add(vector_label)

        # Matrix multiplication displayed bottom-left
        multiplication = MathTex(
            r"A \vec{v} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}"
            r"\begin{bmatrix} 2 \\ 1 \end{bmatrix}"
            r"= \begin{bmatrix} 3 \\ 1 \end{bmatrix}"
        ).scale(0.7).to_corner(DL)
        self.play(Write(multiplication))
        self.wait(1)

        # Apply matrix transformation
        self.apply_matrix(matrix)
        # Update vector label to A·v
        new_label = MathTex(r"A\vec{v} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}")
        new_label.next_to(vector.get_end(), RIGHT + UP * 0.2)

        self.play(Transform(vector_label, new_label))

        self.wait(0.5)

        

        self.wait(2)
'''