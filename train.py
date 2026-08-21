"""
train.py
--------
Mini ML Project: Apple Quality Classification
Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data

ทำหน้าที่:
1. โหลดและทำความสะอาดข้อมูล
2. สำรวจข้อมูล (EDA) แบบสรุปสั้น ๆ
3. เทรนโมเดลหลายตัว เปรียบเทียบ แล้วเลือกตัวที่ดีที่สุด
4. บันทึกโมเดล + scaler ไว้ใช้ใน Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = Path("data/apple_quality.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "Size",
    "Weight",
    "Sweetness",
    "Crunchiness",
    "Juiciness",
    "Ripeness",
    "Acidity",
]
TARGET_COL = "Quality"


def load_and_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ไฟล์จาก Kaggle มีแถวสุดท้ายเป็น metadata (Created_by_...) ที่ค่าอื่น ๆ เป็น NaN หมด -> ตัดทิ้ง
    df = df.dropna(subset=["A_id"]).copy()

    # คอลัมน์ Acidity บางทีถูกอ่านเป็น object เพราะแถว metadata ปนมา -> แปลงเป็นตัวเลข
    df["Acidity"] = pd.to_numeric(df["Acidity"], errors="coerce")

    # ตัดคอลัมน์ id ทิ้ง ไม่ใช่ feature ที่มีความหมายต่อคุณภาพ
    df = df.drop(columns=["A_id"])

    # เผื่อมี NaN หลุดมา ตัดทิ้งให้สะอาด
    df = df.dropna()

    return df.reset_index(drop=True)


def main():
    print("== 1. โหลดและทำความสะอาดข้อมูล ==")
    df = load_and_clean_data(DATA_PATH)
    print(f"จำนวนแถวหลังทำความสะอาด: {len(df)}")
    print(df.dtypes)
    print(df[TARGET_COL].value_counts())

    X = df[FEATURE_COLS]
    y_raw = df[TARGET_COL]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)  # bad=0, good=1 (เรียงตามตัวอักษร)
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    print("\n== 2. แบ่งข้อมูล train/test ==")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n== 3. เทรนและเปรียบเทียบโมเดล ==")
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42
        ),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="accuracy"
        )
        model.fit(X_train_scaled, y_train)
        test_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        results[name] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_acc": test_acc,
        }
        print(
            f"{name:20s} | CV acc: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
            f"| Test acc: {test_acc:.4f}"
        )

    best_name = max(results, key=lambda k: results[k]["test_acc"])
    best_model = results[best_name]["model"]
    print(f"\n== เลือกโมเดลที่ดีที่สุด: {best_name} ==")

    y_pred = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(
            best_model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        print("\nFeature importance:")
        print(importances)

    print("\n== 4. บันทึกโมเดล ==")
    joblib.dump(best_model, MODEL_DIR / "apple_quality_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(FEATURE_COLS, MODEL_DIR / "feature_cols.pkl")

    meta = {
        "best_model_name": best_name,
        "test_accuracy": results[best_name]["test_acc"],
        "cv_accuracy_mean": results[best_name]["cv_mean"],
        "feature_ranges": {
            col: (float(X[col].min()), float(X[col].max())) for col in FEATURE_COLS
        },
    }
    joblib.dump(meta, MODEL_DIR / "meta.pkl")

    print("บันทึกโมเดลและไฟล์ที่เกี่ยวข้องไว้ที่โฟลเดอร์ model/ เรียบร้อย")


if __name__ == "__main__":
    main()
