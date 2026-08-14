import cv2
import numpy as np
import time
import os
import sys

from neuralnetwork import neuralnetwork


# =========================
# 路径
# =========================

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(FILE_DIR)

sys.path.insert(0, FILE_DIR)


# =========================
# 模型
# =========================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "mnist.model"
)

n = neuralnetwork.load(MODEL_PATH)


# =========================
# 参数
# =========================

SIZE = 400
PANEL_WIDTH = 220


# =========================
# 画布
# =========================

canvas = np.zeros(
    (SIZE, SIZE),
    dtype=np.uint8
)


drawing = False
last_point = None

last_draw_time = 0

result = ""

# 当前数字的概率
probabilities = None

# 当前数字的 MNIST 图像
digit_image = None


# =========================
# 鼠标绘制
# =========================

def draw(event, x, y, flags, param):

    global drawing
    global last_point
    global last_draw_time
    x -= PANEL_WIDTH

    if x < 0 or x >= SIZE or y < 0 or y >= SIZE:
        return
    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        last_point = (x, y)

        cv2.circle(
            canvas,
            (x, y),
            12,
            255,
            -1
        )

        last_draw_time = time.time()


    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            cv2.line(
                canvas,
                last_point,
                (x, y),
                255,
                20
            )

            last_point = (x, y)

            last_draw_time = time.time()


    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False
        last_point = None

        last_draw_time = time.time()


# =========================
# MNIST 预处理
# =========================

def preprocess(img):

    _, th = cv2.threshold(
        img,
        30,
        255,
        cv2.THRESH_BINARY
    )

    # 补断点
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    th = cv2.dilate(
        th,
        kernel,
        iterations=1
    )

    ys, xs = np.where(
        th > 0
    )

    if len(xs) == 0:
        return None, None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop = th[
        y1:y2 + 1,
        x1:x2 + 1
    ]

    h, w = crop.shape

    scale = 20 / max(h, w)

    nw = max(
        1,
        int(w * scale)
    )

    nh = max(
        1,
        int(h * scale)
    )

    crop = cv2.resize(
        crop,
        (nw, nh)
    )

    img28 = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x = (28 - nw) // 2
    y = (28 - nh) // 2

    img28[
        y:y + nh,
        x:x + nw
    ] = crop

    data = img28.reshape(784)

    data = (
        data / 255.0 * 0.99
    ) + 0.01

    return data, img28


# =========================
# 识别单个数字
# =========================

def recognize():

    global probabilities
    global digit_image

    data, img28 = preprocess(canvas)

    if data is None:

        probabilities = None
        digit_image = None

        return ""

    output = n.query(data)

    number = np.argmax(output)

    probabilities = output.copy()

    digit_image = img28.copy()

    return str(number)


# =========================
# 创建窗口
# =========================

WINDOW_WIDTH = PANEL_WIDTH + SIZE

cv2.namedWindow(
    "number"
)

cv2.setMouseCallback(
    "number",
    draw
)


# =========================
# 主循环
# =========================

while True:

    # 整个窗口
    show = np.zeros(
        (SIZE, WINDOW_WIDTH, 3),
        dtype=np.uint8
    )


    # =====================
    # 右侧：手写区域
    # =====================

    show[:, PANEL_WIDTH:] = cv2.cvtColor(
        canvas,
        cv2.COLOR_GRAY2BGR
    )


    # =====================
    # 左侧：数字图像
    # =====================

    if digit_image is not None:

        digit_show = cv2.resize(
            digit_image,
            (120, 120),
            interpolation=cv2.INTER_NEAREST
        )

        digit_show = cv2.cvtColor(
            digit_show,
            cv2.COLOR_GRAY2BGR
        )

        show[
            20:140,
            50:170
        ] = digit_show


    # =====================
    # 左侧：概率
    # =====================

    if probabilities is not None:

        for i in range(10):

            p = float(
                probabilities[i]
            )

            text = "{}: {:.2f}%".format(
                i,
                p * 100
            )

            cv2.putText(
                show,
                text,
                (20, 175 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1
            )


    # =====================
    # 右上：识别结果
    # =====================

    cv2.putText(
        show,
        "Result: " + result,
        (PANEL_WIDTH + 20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )


    # =====================
    # 显示
    # =====================

    cv2.imshow(
        "number",
        show
    )


    # =====================
    # 停顿 300ms 自动识别
    # =====================

    if (
        np.any(canvas)
        and time.time() - last_draw_time > 0.3
    ):

        result = recognize()

        # 防止一直重复识别
        last_draw_time = time.time()


    # =====================
    # 键盘
    # =====================

    key = cv2.waitKey(1)


    # ESC
    if key == 27:
        break


    # X 清屏
    if key == ord('x'):

        canvas[:] = 0

        result = ""

        probabilities = None

        digit_image = None


cv2.destroyAllWindows()