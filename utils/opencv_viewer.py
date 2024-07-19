from typing import List, Tuple, Union

import cv2
import numpy as np


class View:
    def __init__(
        self, width: int, height: int, ortho_size=(10, 10), center=(0, 0), angle=0.0
    ):
        self.width = width
        self.height = height
        self.ortho_width, self.ortho_height = ortho_size
        self.center_x, self.center_y = center
        self.angle = angle

        # 预计算旋转矩阵
        angle_rad = self.angle
        self.rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

    def build_frame(self):
        return np.full([self.height, self.width, 3], 0, dtype=np.uint8)

    def convert_space_to_uv(self, x: Union[float, np.array], y: Union[float, np.array]):
        """Convert space coordinates to UV coordinates."""
        # Translate coordinates
        x = x - self.center_x
        y = y - self.center_y

        # Rotate coordinates using the precomputed rotation matrix
        vec = np.array([x, y])
        x_rot, y_rot = np.dot(self.rotation_matrix, vec)

        # Convert to UV
        u = (x_rot + self.ortho_width / 2) / self.ortho_width
        v = (y_rot + self.ortho_height / 2) / self.ortho_height
        return u, v

    def convert_uv_to_pixel(self, u, v):
        """Convert UV coordinates (normalized) to pixel coordinates."""
        x = int(u * self.width)
        y = int((1 - v) * self.height)  # Invert y for pixel coordinates
        return x, y

    def convert_uv_array_to_pixel(self, u: np.ndarray, v: np.ndarray):
        x = (u * self.width).astype(np.int32)
        y = ((1 - v) * self.height).astype(np.int32)
        return x, y

    def draw_line(self, frame, start_space, end_space, color=(255, 0, 0), thickness=2):
        start_uv = self.convert_space_to_uv(*start_space)
        end_uv = self.convert_space_to_uv(*end_space)
        start_point = self.convert_uv_to_pixel(*start_uv)
        end_point = self.convert_uv_to_pixel(*end_uv)
        cv2.line(frame, start_point, end_point, color, thickness)

    def draw_polyline(self, frame, x, y, color=(0, 255, 0), thickness=2):
        points = np.array(
            self.convert_uv_array_to_pixel(*self.convert_space_to_uv(x, y)),
            dtype=np.int32,
        ).T

        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness)

    def draw_rectangle(
        self, frame, top_left_space, bottom_right_space, color=(0, 255, 0), thickness=2
    ):
        top_left_uv = self.convert_space_to_uv(*top_left_space)
        bottom_right_uv = self.convert_space_to_uv(*bottom_right_space)
        top_left = self.convert_uv_to_pixel(*top_left_uv)
        bottom_right = self.convert_uv_to_pixel(*bottom_right_uv)
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)

    def draw_circle(
        self, frame, center_space, radius_space, color=(0, 0, 255), thickness=2
    ):
        center_uv = self.convert_space_to_uv(*center_space)
        center = self.convert_uv_to_pixel(*center_uv)
        radius = int(
            radius_space
            * min(self.width, self.height)
            / max(self.ortho_width, self.ortho_height)
        )
        cv2.circle(frame, center, radius, color, thickness)

    def draw_text(
        self,
        frame,
        text,
        position_space,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1,
        color=(255, 255, 255),
        thickness=2,
    ):
        position_uv = self.convert_space_to_uv(*position_space)
        position = self.convert_uv_to_pixel(*position_uv)
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

    def draw_rotated_rectangle(
        self, frame, center_space, size, angle, color=(0, 255, 255), thickness=2
    ):
        cx, cy = center_space
        w, h = size
        angle_rad = angle
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        # Define the four corners of the rectangle
        corners = np.array(
            [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
        )

        # Rotate the corners and translate to the center
        rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([cx, cy])

        # Convert to UV and then to pixel coordinates
        points = [
            self.convert_uv_to_pixel(*self.convert_space_to_uv(x, y))
            for x, y in rotated_corners
        ]
        points = np.array(points, dtype=np.int32)

        # Draw the rotated rectangle
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)


if __name__ == "__main__":
    view = View(1600, 800, (160, 80), center=(0, 0), angle=0.0)

    frame = view.build_frame()
    print("???")
    print(frame.shape)

    view.draw_line(frame, (-20, -10), (30, 40))
    view.draw_circle(frame, (0, 0), 20)
    view.draw_rotated_rectangle(frame, (10, 0), (20, 5), np.pi / 3)
    view.draw_text(frame, "Test", (10, 10))
    view.draw_polyline(
        frame, np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 1.0, 0.0])
    )

    ret = cv2.imwrite("./test.png", frame)

    print(ret)
