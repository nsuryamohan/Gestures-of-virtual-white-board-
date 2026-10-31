import cv2
import numpy as np
import HandTrackingModule as htm

# ============================================================
# CONFIGURATION
# ============================================================

width, height = 1280, 720

brushThickness = 15

drawColor = (255, 0, 255)  # Purple

xp, yp = 0, 0


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(3, width)
cap.set(4, height)


# ============================================================
# HAND DETECTOR
# ============================================================

detector = htm.handDetector(
    detectionCon=0.85,
    maxHands=1
)


# ============================================================
# WHITE CANVAS
# ============================================================

imgCanvas = np.ones(
    (height, width, 3),
    dtype=np.uint8
) * 255


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, camera = cap.read()

    if not success:
        print("Camera could not be accessed.")
        break

    camera = cv2.flip(camera, 1)


    # ========================================================
    # HAND DETECTION
    # ========================================================

    detector.findHands(camera, draw=False)

    lmList = detector.findPosition(
        camera,
        draw=False
    )


    # ========================================================
    # CREATE WHITE DISPLAY
    # ========================================================

    display = np.ones(
        (height, width, 3),
        dtype=np.uint8
    ) * 255


    # ========================================================
    # GET HAND INFORMATION
    # ========================================================

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()


        # ====================================================
        # DRAW EXISTING DRAWING
        # ====================================================

        # Create mask for non-white pixels
        canvasGray = cv2.cvtColor(
            imgCanvas,
            cv2.COLOR_BGR2GRAY
        )

        drawingMask = cv2.inRange(
            canvasGray,
            0,
            254
        )

        # Copy only drawing pixels
        display[drawingMask > 0] = imgCanvas[drawingMask > 0]


        # ====================================================
        # SELECTION MODE
        # ====================================================

        if fingers[1] and fingers[2]:

            xp, yp = 0, 0

            if y1 < 125:

                # RED
                if 250 < x1 < 450:

                    drawColor = (0, 0, 255)

                # BLUE
                elif 550 < x1 < 750:

                    drawColor = (255, 0, 0)

                # GREEN
                elif 800 < x1 < 950:

                    drawColor = (0, 255, 0)

                # CLEAR
                elif 1050 < x1 < 1200:

                    imgCanvas = np.ones(
                        (height, width, 3),
                        dtype=np.uint8
                    ) * 255


            # Selection box
            cv2.rectangle(
                display,
                (x1, y1 - 25),
                (x2, y2 + 25),
                drawColor,
                cv2.FILLED
            )


        # ====================================================
        # DRAWING MODE
        # ====================================================

        elif fingers[1] and not fingers[2]:

            # Drawing cursor
            cv2.circle(
                display,
                (x1, y1),
                12,
                drawColor,
                cv2.FILLED
            )


            # First point
            if xp == 0 and yp == 0:

                xp, yp = x1, y1


            # Draw on white canvas
            cv2.line(
                imgCanvas,
                (xp, yp),
                (x1, y1),
                drawColor,
                brushThickness
            )


            xp, yp = x1, y1


        else:

            xp, yp = 0, 0


        # ====================================================
        # DRAW HAND LANDMARKS
        # ====================================================

        for landmark in lmList:

            landmark_id, x, y = landmark

            cv2.circle(
                display,
                (x, y),
                5,
                (0, 0, 0),
                cv2.FILLED
            )


        # ====================================================
        # HAND CONNECTIONS
        # ====================================================

        connections = [

            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            (0, 17)
        ]


        for start, end in connections:

            x_start = lmList[start][1]
            y_start = lmList[start][2]

            x_end = lmList[end][1]
            y_end = lmList[end][2]

            cv2.line(
                display,
                (x_start, y_start),
                (x_end, y_end),
                (0, 0, 0),
                2
            )


    # ========================================================
    # TOOLBAR
    # ========================================================

    cv2.rectangle(
        display,
        (0, 0),
        (width, 125),
        (40, 40, 40),
        cv2.FILLED
    )


    cv2.putText(
        display,
        "RED",
        (300, 75),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 0, 255),
        3
    )


    cv2.putText(
        display,
        "BLUE",
        (600, 75),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        3
    )


    cv2.putText(
        display,
        "GREEN",
        (850, 75),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 255, 0),
        3
    )


    cv2.putText(
        display,
        "CLEAR",
        (1050, 75),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WHITEBOARD BORDER
    # ========================================================

    cv2.rectangle(
        display,
        (5, 130),
        (1275, 715),
        (0, 0, 0),
        4
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Gesture Virtual Whiteboard",
        display
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()