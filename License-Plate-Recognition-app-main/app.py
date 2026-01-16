import sys
import streamlit as st

st.title("Debug Check")
st.write("Python version:", sys.version)
st.success("App booted successfully")

import easyocr
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import tempfile
import os
import numpy as np
import re
import cv2


DIGIT_TO_LETTER = {
    "0": "O",
    "1": "T",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G",
}

LETTER_TO_DIGIT = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "Q": "0",
    "L": "1",
}


def fix_to_letter(ch: str) -> str:
    return DIGIT_TO_LETTER.get(ch, ch)


def fix_to_digit(ch: str) -> str:
    return LETTER_TO_DIGIT.get(ch, ch)

def clean_plate(plate: str) -> str:
    if not plate:
        return ""

    p = plate.upper().replace("IND", "")
    p = re.sub(r'[^A-Z0-9]', '', p)

    if len(p) < 6:
        return p
    chars = list(p)

    # State code (letters)
    for i in range(0, 2):
        chars[i] = fix_to_letter(chars[i])

    # RTO code (digits)
    for i in range(2, 4):
        chars[i] = fix_to_digit(chars[i])

    # Series (letters, 1–3 slots)
    for i in range(4, 6):
        chars[i] = fix_to_letter(chars[i])

    # Last 4 digits
    for i in range(6,len(chars)):
        chars[i] = fix_to_digit(chars[i])
    
    if chars[0].isalpha() and chars[1].isalpha() and chars[2].isalpha(): 
        chars=chars[1:]
    if chars[-1].isdigit() and chars[-2].isdigit() and chars[-3].isdigit() and chars[-4].isdigit() and chars[-5].isdigit():
        chars=chars[:-1] 

    return "".join(chars)

INDIAN_PLATE_FORMAT = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')

def is_valid_indian_plate(plate: str) -> bool:
    plate = plate.replace(" ", "").upper()
    return bool(INDIAN_PLATE_FORMAT.match(plate))
# ---------------------------------------------------------

# Page Setup
st.set_page_config(page_title="Indian LPR Pro", layout="wide")
st.title("Real-Time Indian License Plate Recognition")
st.markdown("**YOLOv11 + EasyOCR • Live Webcam • Image • Video • CSV Export**")

# Load Models 
@st.cache_resource
def load_models():
    model = YOLO('best.pt')
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

model, reader = load_models()

# Session state
if 'plates' not in st.session_state:
    st.session_state.plates = []
if 'running' not in st.session_state:
    st.session_state.running = False

# Tabs 
tab1, tab2= st.tabs(["Upload Image", "Upload Video"])

# TAB 1: UPLOAD IMAGE 

with tab1:
    st.markdown("### Upload Image")
    uploaded_img = st.file_uploader("Drop an image of a vehicle", type=["jpg", "jpeg", "png"])

    if uploaded_img and st.button("Detect Plates in Image", type="primary"):
        file_bytes = uploaded_img.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img, conf=0.15, iou=0.5, max_det=10, imgsz=640, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate = img[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            plate = cv2.resize(plate, (240, 80))
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            text = reader.readtext(gray, detail=0, paragraph=False,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

            # CLEAN + VALIDATE
            plate_no = clean_plate("".join(text))
            if is_valid_indian_plate(plate_no):
                if plate_no not in [p["Plate"] for p in st.session_state.plates]:
                    st.session_state.plates.append(
                        {"Time": datetime.now().strftime("%H:%M:%S"), "Plate": plate_no}
                    )

            # DRAW
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
            cv2.putText(img, plate_no, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")
        st.success("Image processed!")


# TAB 2: UPLOAD VIDEO 

with tab2:
    st.markdown("### Upload Video")
    uploaded_video = st.file_uploader("Drop a video file", type=["mp4", "avi", "mov"])

    if uploaded_video and st.button("Process Video & Track Plates", type="primary"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        progress = st.progress(0)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.15, iou=0.5, max_det=10, imgsz=640, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate = frame[y1:y2, x1:x2]
                if plate.size == 0:
                    continue

                plate = cv2.resize(plate, (240, 80))
                gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                text = reader.readtext(gray, detail=0, paragraph=False,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

                # CLEAN + VALIDATE
                plate_no = clean_plate("".join(text))
                if is_valid_indian_plate(plate_no):
                    if plate_no not in [p["Plate"] for p in st.session_state.plates]:
                        st.session_state.plates.append(
                            {"Time": f"Video {current}", "Plate": plate_no}
                        )

                # DRAW
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                cv2.putText(frame, plate_no, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            current += 1
            progress.progress(current / frame_count)

        cap.release()
        os.unlink(video_path)
        st.success("Video processed completely!")

# RESULTS & DOWNLOAD

st.markdown("## All Detected Plates")

col_table, col_buttons = st.columns([4, 1])

with col_table:
    if st.session_state.plates:
        df = pd.DataFrame(st.session_state.plates)
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("No plates detected yet — try webcam, image, or video!")

with col_buttons:
    st.markdown("### Actions")
    
    if st.session_state.plates:
        df = pd.DataFrame(st.session_state.plates)
        csv = df.to_csv(index=False).encode()

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"Indian_LPR_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Delete All Plates", type="secondary", use_container_width=True):
            st.session_state.plates = []
            st.success("All plates cleared!")
            st.rerun()

    st.markdown(f"**Total Detected:** `{len(st.session_state.plates)}`")

