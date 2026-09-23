import io
from datetime import datetime

import folium
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from streamlit_folium import st_folium
from ultralytics import YOLO

st.set_page_config(
    page_title="AquaMap KG",
    page_icon="🌊",
    layout="wide",
)

MODEL_REPO = "FathomNet/trash-detector"
MODEL_FILE = (
    "trash_mbari_09072023_640imgsz_"
    "50epochs_yolov8.pt"
)


@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )
    return YOLO(path)


if "findings" not in st.session_state:
    st.session_state.findings = []

st.title("AquaMap KG")
st.caption(
    "Поиск подводного мусора "
    "и планирование уборок"
)

with st.sidebar:
    st.header("Данные точки")
    lake = st.text_input(
        "Озеро или водоём",
        "Иссык-Куль",
    )
    latitude = st.number_input(
        "Широта",
        value=42.4400,
        format="%.6f",
    )
    longitude = st.number_input(
        "Долгота",
        value=77.2500,
        format="%.6f",
    )
    depth = st.number_input(
        "Глубина, м",
        min_value=0.0,
        value=2.0,
        step=0.5,
    )
    confidence = st.slider(
        "Порог уверенности",
        0.05,
        0.90,
        0.20,
        0.05,
    )
    notes = st.text_area("Комментарий")

uploaded = st.file_uploader(
    "Загрузите подводное фото",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.subheader("Исходное фото")
        st.image(image, use_container_width=True)

    with st.spinner("Ищем мусор..."):
        model = load_model()
        result = model.predict(
        image,
        conf=confidence,
        imgsz=1280,
        verbose=False,
    )[0]

    names = result.names
    detected = []
    if result.boxes is not None:
        for class_id in result.boxes.cls.tolist():
            detected.append(names[int(class_id)])

    trash_count = sum(
        1
        for name in detected
        if "trash" in name.lower()
    )

    with right:
        st.subheader("Результат")
        annotated = result.plot()
        annotated = annotated[:, :, ::-1]
        st.image(
            annotated,
            use_container_width=True,
        )

    if trash_count:
        st.success(
            f"Обнаружено объектов мусора: "
            f"{trash_count}"
        )
    else:
        st.warning(
            "Мусор не найден. Это не "
            "гарантирует чистоту участка."
        )

    if detected:
        st.write(
            "Все распознанные классы:",
            ", ".join(detected),
        )

    if st.button("Добавить точку на карту"):
        st.session_state.findings.append(
            {
                "date": datetime.now().isoformat(
                    timespec="minutes"
                ),
                "lake": lake,
                "latitude": latitude,
                "longitude": longitude,
                "depth_m": depth,
                "trash_count": trash_count,
                "status": "Найдено",
                "notes": notes,
            }
        )
        st.success("Точка добавлена")

st.subheader("Карта находок")
map_center = [42.4400, 77.2500]
if st.session_state.findings:
    last = st.session_state.findings[-1]
    map_center = [
        last["latitude"],
        last["longitude"],
    ]

map_view = folium.Map(
    location=map_center,
    zoom_start=8,
    tiles="OpenStreetMap",
)

for item in st.session_state.findings:
    popup = (
        f"<b>{item['lake']}</b><br>"
        f"Мусор: {item['trash_count']}<br>"
        f"Глубина: {item['depth_m']} м<br>"
        f"Статус: {item['status']}"
    )
    folium.Marker(
        [item["latitude"], item["longitude"]],
        popup=popup,
        tooltip=item["lake"],
    ).add_to(map_view)

st_folium(
    map_view,
    width=None,
    height=460,
    returned_objects=[],
)

if st.session_state.findings:
    table = pd.DataFrame(
        st.session_state.findings
    )
    st.subheader("Журнал уборок")
    st.dataframe(
        table,
        use_container_width=True,
    )
    csv_data = table.to_csv(
        index=False
    ).encode("utf-8")
    st.download_button(
        "Скачать CSV",
        data=csv_data,
        file_name="aquamapkg_findings.csv",
        mime="text/csv",
    )

st.caption(
    "Прототип: результат модели требует "
    "проверки человеком. Координаты "
    "в демо вводятся вручную."
)
