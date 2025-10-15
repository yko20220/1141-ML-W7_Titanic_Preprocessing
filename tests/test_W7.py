# tests/test_W7.py
# W7 Titanic Preprocessing 自動評分測試

import pytest
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
import os
import re
import types

# -------------------------
# 取得學生提交程式
# -------------------------
SUBMIT_DIR = Path(__file__).parent.parent / "submit"
student_files = list(SUBMIT_DIR.glob("*.py"))
if not student_files:
    raise FileNotFoundError(f"{SUBMIT_DIR} 沒有學生提交檔案")

student_file = student_files[0]
spec = importlib.util.spec_from_file_location("student_submission", student_file)
student_submission = importlib.util.module_from_spec(spec)

load_error = None
try:
    spec.loader.exec_module(student_submission)
except Exception as e:
    load_error = str(e)
    print(f"❌ 無法載入學生程式 ({student_file.name})：{e}")
    # 建立空殼物件防止中斷
    student_submission = types.SimpleNamespace(
        load_data=lambda *a, **k: (pd.DataFrame(), 0),
        handle_missing=lambda df: df,
        remove_outliers=lambda df: df,
        encode_features=lambda df: df,
        scale_features=lambda df: df,
        split_data=lambda df: (pd.DataFrame(), pd.DataFrame(), [], []),
        save_data=lambda df, path: pd.DataFrame().to_csv(path)
    )

# 匯入學生函式
load_data = getattr(student_submission, "load_data", lambda *a, **k: (pd.DataFrame(), 0))
handle_missing = getattr(student_submission, "handle_missing", lambda df: df)
remove_outliers = getattr(student_submission, "remove_outliers", lambda df: df)
encode_features = getattr(student_submission, "encode_features", lambda df: df)
scale_features = getattr(student_submission, "scale_features", lambda df: df)
split_data = getattr(student_submission, "split_data", lambda df: (pd.DataFrame(), pd.DataFrame(), [], []))
save_data = getattr(student_submission, "save_data", lambda df, path: pd.DataFrame().to_csv(path))

DATA_PATH = "data/titanic.csv"

# -------------------------
# 評分設定
# -------------------------
results = []
POINTS = {
    "程式可執行": 10,
    "載入資料正確": 10,
    "缺失值已處理": 10,
    "異常值已移除": 10,
    "One-hot 編碼正確": 10,
    "標準化正確": 10,
    "資料切割比例正確": 10,
    "輸出檔案存在": 10,
    "輸出欄位一致": 10
}


def check(name, condition, msg=""):
    if condition:
        results.append(f"✅ {name} (+{POINTS.get(name, 0)})")
    else:
        results.append(f"❌ {name} - {msg} (+0)")


def calculate_score():
    score = 0
    for line in results:
        if line.startswith("✅"):
            m = re.search(r"\+(\d+)", line)
            if m:
                score += int(m.group(1))
    return score


def save_results_md(filename="test_results/results.md"):
    score = calculate_score()
    os.makedirs(Path(filename).parent, exist_ok=True)
    content = f"### 🧩 W7 Titanic 前處理作業測試結果\n\n總分: {score}\n\n" + "\n".join(results)
    if load_error:
        content = f"⚠️ **程式執行錯誤：** {load_error}\n\n" + content
    print(content)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


# -------------------------
# 功能測試
# -------------------------

def test_loadable():
    cond = load_error is None
    check("程式可執行", cond, "語法錯誤或模組載入失敗")


def test_load_data():
    df, missing = load_data(DATA_PATH)
    cond = isinstance(df, pd.DataFrame) and "Survived" in df.columns and isinstance(missing, (int, np.integer))
    check("載入資料正確", cond, "未正確載入或缺少必要欄位")


def test_handle_missing():
    df, _ = load_data(DATA_PATH)
    df = handle_missing(df)
    if "Age" in df.columns and "Embarked" in df.columns:
        cond = df["Age"].isnull().sum() == 0 and df["Embarked"].isnull().sum() == 0
    else:
        cond = False
    check("缺失值已處理", cond, "Age 或 Embarked 仍有缺失值")


def test_remove_outliers():
    df, _ = load_data(DATA_PATH)
    df = handle_missing(df)
    df = remove_outliers(df)
    if "Fare" in df.columns:
        mean, std = df["Fare"].mean(), df["Fare"].std()
        cond = df["Fare"].max() <= mean + 3 * std
    else:
        cond = False
    check("異常值已移除", cond, "Fare 未正確移除異常值")


def test_encode_features():
    df, _ = load_data(DATA_PATH)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = encode_features(df)
    cols = df.columns
    expected = ["Sex_female", "Sex_male", "Embarked_S"]
    cond = all(c in cols for c in expected)
    check("One-hot 編碼正確", cond, f"缺少欄位: {expected}")


def test_scale_features():
    df, _ = load_data(DATA_PATH)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    if "Age" in df.columns and "Fare" in df.columns:
        cond = abs(df["Age"].mean()) < 1e-6 and abs(df["Fare"].mean()) < 1e-6
    else:
        cond = False
    check("標準化正確", cond, "Age 或 Fare 未標準化")


def test_split_data():
    df, _ = load_data(DATA_PATH)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    try:
        X_train, X_test, y_train, y_test = split_data(df)
        total = len(X_train) + len(X_test)
        cond = (
            total > 0 and
            abs(len(X_train) / total - 0.8) < 0.05 and
            len(y_train) == len(X_train)
        )
    except Exception:
        cond = False
    check("資料切割比例正確", cond, "比例或長度錯誤")


def test_save_data(tmp_path):
    df, _ = load_data(DATA_PATH)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    output = tmp_path / "titanic_out.csv"
    try:
        save_data(df, output)
        cond_exist = output.exists()
    except Exception:
        cond_exist = False
    check("輸出檔案存在", cond_exist, "CSV 檔案未生成")
    if cond_exist:
        df_out = pd.read_csv(output)
        cond_cols = set(df.columns) <= set(df_out.columns)
        check("輸出欄位一致", cond_cols, "輸出欄位與原始資料不一致")


def test_generate_md():
    save_results_md("test_results/results.md")
