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
import traceback

# -------------------------
# 取得學生提交程式 (Safe Import)
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

# 匯入學生函式（若不存在就補空的）
def safe_getattr(obj, name, default):
    return getattr(obj, name, default)

load_data = safe_getattr(student_submission, "load_data", lambda *a, **k: (pd.DataFrame(), 0))
handle_missing = safe_getattr(student_submission, "handle_missing", lambda df: df)
remove_outliers = safe_getattr(student_submission, "remove_outliers", lambda df: df)
encode_features = safe_getattr(student_submission, "encode_features", lambda df: df)
scale_features = safe_getattr(student_submission, "scale_features", lambda df: df)
split_data = safe_getattr(student_submission, "split_data", lambda df: (pd.DataFrame(), pd.DataFrame(), [], []))
save_data = safe_getattr(student_submission, "save_data", lambda df, path: pd.DataFrame().to_csv(path))

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

def check(name, func, msg=""):
    """統一包測試，避免例外導致整個 CI fail"""
    try:
        result = func()
        if result:
            results.append(f"✅ {name} (+{POINTS.get(name, 0)})")
        else:
            results.append(f"❌ {name} - {msg} (+0)")
    except Exception as e:
        err = f"{e.__class__.__name__}: {e}"
        results.append(f"❌ {name} - 測試發生錯誤 ({err}) (+0)")
        traceback.print_exc()

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
# 功能測試 (全都包進 check)
# -------------------------
def test_all():
    # 1) 可執行
    check("程式可執行", lambda: load_error is None, "語法錯誤或模組載入失敗")

    # 2) 載入資料正確
    def _load_ok():
        df, m = load_data(DATA_PATH)
        return isinstance(df, pd.DataFrame) and "Survived" in df.columns
    check("載入資料正確", _load_ok, "未正確載入資料或缺少必要欄位")

    # 3) 缺失值已處理
    def _missing_ok():
        df, _ = load_data(DATA_PATH)
        df = handle_missing(df)
        return ("Age" in df.columns and df["Age"].isnull().sum() == 0) and \
               ("Embarked" in df.columns and df["Embarked"].isnull().sum() == 0)
    check("缺失值已處理", _missing_ok, "Age 或 Embarked 仍有缺失值")

    # 4) 異常值已移除
    def _outliers_ok():
        df, _ = load_data(DATA_PATH)
        df = handle_missing(df)
        df = remove_outliers(df)
        if "Fare" not in df.columns or len(df) == 0:
            return False
        mean, std = df["Fare"].mean(), df["Fare"].std()
        return df["Fare"].max() <= mean + 3 * std
    check("異常值已移除", _outliers_ok, "Fare 未正確移除異常值")

    # 5) One-hot 編碼正確
    def _encode_ok():
        df, _ = load_data(DATA_PATH)
        df = handle_missing(df)
        df = remove_outliers(df)
        df = encode_features(df)
        expected = ["Sex_female", "Sex_male", "Embarked_S"]
        return all(c in df.columns for c in expected)
    check("One-hot 編碼正確", _encode_ok, "缺少 One-hot 欄位 (需含 Sex_female, Sex_male, Embarked_S)")

    # 6) 標準化正確
    def _scale_ok():
        df, _ = load_data(DATA_PATH)
        df = handle_missing(df)
        df = remove_outliers(df)
        df = encode_features(df)
        df = scale_features(df)
        if "Age" not in df.columns or "Fare" not in df.columns or len(df) == 0:
            return False
        return abs(df["Age"].mean()) < 1e-6 and abs(df["Fare"].mean()) < 1e-6
    check("標準化正確", _scale_ok, "Age 或 Fare 未標準化")

    # 7) 資料切割比例正確
    def _split_ok():
        df, _ = load_data(DATA_PATH)
        df = handle_missing(df)
        df = remove_outliers(df)
        df = encode_features(df)
        df = scale_features(df)
        X_train, X_test, y_train, y_test = split_data(df)
        total = len(X_train) + len(X_test)
        if total == 0:
            return False
        ratio_ok = abs(len(X_train) / total - 0.8) < 0.05
        length_ok = len(y_train) == len(X_train) and len(y_test) == len(X_test)
        return ratio_ok and length_ok
    check("資料切割比例正確", _split_ok, "比例或長度錯誤")

    # 8) 輸出檔案存在
    def _save_exist_ok():
        df, _ = load_data(DATA_PATH)
        df = handle_missing(df)
        df = remove_outliers(df)
        df = encode_features(df)
        df = scale_features(df)
        out = Path("test_results/tmp.csv")
        os.makedirs(out.parent, exist_ok=True)
        try:
            save_data(df, out)
        except Exception:
            return False
        return out.exists()
    check("輸出檔案存在", _save_exist_ok, "CSV 未生成")

    # 9) 輸出欄位一致
    def _save_cols_ok():
        df, _ = load_data(DATA_PATH)
        df = handle_missing(df)
        df = remove_outliers(df)
        df = encode_features(df)
        df = scale_features(df)
        out = Path("test_results/tmp.csv")
        if not out.exists():
            return False
        try:
            df_out = pd.read_csv(out)
        except Exception:
            return False
        return set(df.columns) <= set(df_out.columns)
    check("輸出欄位一致", _save_cols_ok, "輸出欄位與原始資料不一致")

    # 最後輸出報告
    save_results_md("test_results/results.md")
