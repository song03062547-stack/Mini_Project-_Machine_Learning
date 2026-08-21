# 🍎 Apple Quality Prediction — Mini ML Project

**ผู้จัดทำ:** นายปฐมพงศ์ ชัยสรรค์ · รหัสนักศึกษา 664245039 · 66/44

Mini Machine Learning project ที่ทำนาย**คุณภาพของแอปเปิล** (`good` / `bad`)
จากคุณสมบัติทางกายภาพของผล เช่น ขนาด น้ำหนัก ความหวาน ความกรอบ ความฉ่ำน้ำ
ความสุก และความเป็นกรด พร้อม Web App สำหรับทำนายผลแบบ real-time ด้วย Streamlit

**Dataset:** [Apple Quality — Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data)
โดย Nidula Elgiriyewithana (4000 ตัวอย่าง, 7 features)

## Demo

🔗 Live app: `<ใส่ลิงก์ Streamlit Cloud ของคุณที่นี่หลังจาก deploy เสร็จ>`

## โครงสร้างโปรเจกต์

```
apple-quality-ml/
├── data/
│   └── apple_quality.csv      # ข้อมูลดิบจาก Kaggle
├── model/                      # โมเดลที่เทรนแล้ว (สร้างจาก train.py)
│   ├── apple_quality_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_cols.pkl
│   └── meta.pkl
├── train.py                    # สคริปต์เทรนโมเดล (EDA + train + save)
├── app.py                      # Streamlit web app
├── requirements.txt
└── README.md
```

## Feature ที่ใช้ทำนาย

| Feature | ความหมาย |
|---|---|
| Size | ขนาดของผล (standardized) |
| Weight | น้ำหนักของผล |
| Sweetness | ระดับความหวาน |
| Crunchiness | ความกรอบของเนื้อ |
| Juiciness | ความฉ่ำน้ำ |
| Ripeness | ระดับความสุก |
| Acidity | ความเป็นกรด |

## วิธีรันบนเครื่องตัวเอง (Local)

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/apple-quality-ml.git
cd apple-quality-ml

# 2. สร้าง virtual environment (แนะนำ)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. ติดตั้ง dependencies
pip install -r requirements.txt

# 4. เทรนโมเดล (สร้างไฟล์ในโฟลเดอร์ model/)
python train.py

# 5. รัน Streamlit app
streamlit run app.py
```

จากนั้นเปิดเบราว์เซอร์ไปที่ `http://localhost:8501`

## Machine Learning Pipeline

1. **Data Cleaning** — โหลด CSV, ตัดแถว metadata ท้ายไฟล์ทิ้ง, แปลงคอลัมน์ `Acidity`
   ให้เป็นตัวเลข, ตัดคอลัมน์ `A_id` ที่ไม่มีผลต่อการทำนาย
2. **Preprocessing** — Scale ข้อมูลด้วย `StandardScaler`, เข้ารหัส label ด้วย
   `LabelEncoder`
3. **Model Training & Selection** — เทรนและเปรียบเทียบ 3 โมเดลด้วย 5-fold
   cross-validation: Logistic Regression, Random Forest, SVM (RBF kernel)
   แล้วเลือกโมเดลที่แม่นยำที่สุดบน test set
4. **Model Persistence** — บันทึกโมเดล, scaler, label encoder ด้วย `joblib`
5. **Deployment** — โหลดโมเดลเข้า Streamlit app เพื่อทำนายผลแบบ interactive

### ผลลัพธ์โดยสรุป (จากการรันครั้งล่าสุด)

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | ~74.7% | ~74.4% |
| **Random Forest** ✅ | ~88.2% | **~87.6%** |
| SVM (RBF) | ~89.1% | ~87.3% |

โมเดลที่เลือกใช้: **Random Forest** (ความแม่นยำสูงสุดบน test set)

## Deploy ขึ้น Streamlit Community Cloud

1. Push โปรเจกต์นี้ขึ้น GitHub (repo public หรือ private ก็ได้)
2. ไปที่ [share.streamlit.io](https://share.streamlit.io) แล้ว sign in ด้วย GitHub
3. กด **New app** → เลือก repo / branch นี้ → ตั้งค่า **Main file path** เป็น `app.py`
4. กด **Deploy** รอสักครู่ ระบบจะติดตั้ง dependency จาก `requirements.txt`
   และรันแอปให้อัตโนมัติ
5. ได้ลิงก์สาธารณะรูปแบบ `https://<app-name>.streamlit.app` นำไปแปะในหัวข้อ Demo ด้านบนได้เลย

> หมายเหตุ: ไฟล์โมเดลในโฟลเดอร์ `model/` ต้องถูก push ขึ้น GitHub ไปด้วย
> (รัน `python train.py` ก่อน commit เพื่อให้ไฟล์ `.pkl` ถูกสร้างขึ้นมา)

## ผู้จัดทำ

| | |
|---|---|
| ชื่อ-นามสกุล | นายปฐมพงศ์ ชัยสรรค์ |
| รหัสนักศึกษา | 664245039 |
| ที่อยู่ | 66/44 |

## License

MIT — ใช้/ดัดแปลงได้อิสระ ให้เครดิตแหล่งข้อมูลต้นทางตามเงื่อนไขของ Kaggle dataset
