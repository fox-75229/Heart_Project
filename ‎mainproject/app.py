# app.py (v10 - 邏輯迴歸預測版)

from flask import Flask, render_template, jsonify, Response, request
import pandas as pd
import numpy as np
import os
import io # 用於處理圖片 I/O

# 匯入決策樹和 Graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz # 匯入 Graphviz

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
#---Flask 應用設定---
app = Flask(__name__)

# 自定義JSON序列化設定
app.json.ensure_ascii = False

# --- 建立全域快取 (Cache) ---
dt_cache = {}
lr_cache = {} # <-- 【新增】邏輯迴歸的快取

# --- 網頁路由 (不變) ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/decision_tree")
def decision_tree():
    return render_template("decision_tree.html")

@app.route("/logistic")
def logistic():
    return render_template("logistic.html")

# -----------------------------------------------------------------
#   邏輯迴歸 API 
# -----------------------------------------------------------------

def get_logistic_results():
    """
    【【【新增】】】
    這是一個內部函式，負責訓練「邏輯迴歸」模型並快取結果。
    """
    # 檢查快取
    if "model_plot" in lr_cache:
        return lr_cache
    
    print("--- 正在訓練邏輯迴歸模型，請稍候... ---")

    # 1. 載入「已清理」的資料
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'heart.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到檔案:{csv_path}")
    qust_cleaned = pd.read_csv(csv_path)
    
    # 2. [A 部分] 訓練「高分模型」 (All Features)
    target = '是否有心臟病'
    if target not in qust_cleaned.columns:
        raise KeyError(f"CSV 檔案缺少目標欄位: '{target}'")
    X_all = qust_cleaned.drop(columns=[target])
    y_all = qust_cleaned[target]
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    model_best = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    model_best.fit(X_train_all, y_train_all)
    y_pred_class_best = model_best.predict(X_test_all)
    y_pred_proba_best = model_best.predict_proba(X_test_all)[:, 1]
    best_metrics = {
        "accuracy": round(accuracy_score(y_test_all, y_pred_class_best), 4),
        "precision": round(precision_score(y_test_all, y_pred_class_best), 4),
        "recall": round(recall_score(y_test_all, y_pred_class_best), 4),
        "f1": round(f1_score(y_test_all, y_pred_class_best), 4),
        "auc": round(roc_auc_score(y_test_all, y_pred_proba_best), 4)
    }
    
    # 3. [B 部分] 訓練「圖表模型」 (2 Features)
    feature_1 = 'ST 段斜率'
    feature_2 = '運動是否誘發心絞痛'
    if feature_1 not in qust_cleaned.columns or feature_2 not in qust_cleaned.columns:
        raise KeyError(f"qust_cleaned 中缺少 '{feature_1}' 或 '{feature_2}'")
    
    x_plot = qust_cleaned[[feature_1, feature_2]]
    y_plot = qust_cleaned[target]
    
    X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
        x_plot, y_plot, test_size=0.3, random_state=42, stratify=y_plot
    )
    
    model_plot = LogisticRegression(random_state=42, solver='liblinear')
    model_plot.fit(X_train_plot, y_train_plot)
    
    x_min, x_max = x_plot.iloc[:, 0].min() - 0.5, x_plot.iloc[:, 0].max() + 0.5
    y_min, y_max = x_plot.iloc[:, 1].min() - 0.5, x_plot.iloc[:, 1].max() + 0.5
    step = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    meshgrid_data = np.c_[xx.ravel(), yy.ravel()]
    meshgrid_data_df = pd.DataFrame(meshgrid_data, columns=[feature_1, feature_2])
    Z = model_plot.predict(meshgrid_data_df)
    Z = Z.reshape(xx.shape)

    # 4. [C 部分] 組合圖表資料
    chart_data = {
        "train_points":{ "x1": X_train_plot[feature_1].tolist(), "x2": X_train_plot[feature_2].tolist(), "y": y_train_plot.tolist() },
        "test_points":{ "x1": X_test_plot[feature_1].tolist(), "x2": X_test_plot[feature_2].tolist(), "y": y_test_plot.tolist() },
        "decision_boundary":{ "xx": xx.tolist(), "yy": yy.tolist(), "Z": Z.tolist() }
    }
    
    chart_description = {
        "dataset": "心臟衰竭資料集",
        "x1_feature": feature_1,
        "x2_feature": feature_2,
        "total_samples": len(qust_cleaned),
        "train_size": len(X_train_all), # <-- 修正：使用高分模型的樣本數
        "test_size": len(X_test_all),   # <-- 修正：使用高分模型的樣本數
        "target": target,
        "selected_features": X_all.columns.tolist(), # <-- 修正：回傳所有特徵
        "y_min": y_min # <-- 【新增】回傳 Y 軸最小值，用於動畫
    }

    # 5. 儲存到快取
    lr_cache["model_plot"] = model_plot
    lr_cache["best_metrics"] = best_metrics
    lr_cache["chart_data"] = chart_data
    lr_cache["chart_description"] = chart_description
    
    print("--- 邏輯迴歸模型訓練完畢並已快取！ ---")
    return lr_cache


@app.route("/api/logistic/data")
def logistic_data():
    """
    【【【已修改】】】
    改為從快取函式 get_logistic_results() 讀取資料
    """
    try:
        # 1. 從快取中取得 results
        results = get_logistic_results()
        
        # 2. 合併回應
        response = {
            "success": True,
            "data": results["chart_data"],
            "metrics": results["best_metrics"],
            "description": results["chart_description"]
        }
        return jsonify(response)
        
    # 錯誤處理 (不變)
    except FileNotFoundError as e:
        print(f"檔案錯誤: {e}")
        return jsonify({"success": False, "error": str(e)}), 404
    except KeyError as e:
        print(f"欄位錯誤: {e}")
        return jsonify({"success": False, "error": f"CSV 欄位錯誤: {e}"}), 400
    except Exception as e:
        print(f"伺服器錯誤: {e}") 
        return jsonify({"success": False, "error": f"伺服器內部錯誤: {e}"}), 500

# --- 【【【新增 API 路由：邏輯迴歸預測】】】 ---
@app.route("/api/logistic/predict", methods=['POST'])
def logistic_predict():
    """
    使用快取的「圖表模型 (model_plot)」來執行預測
    """
    try:
        # 1. 取得快取的模型
        results = get_logistic_results()
        model_plot = results["model_plot"]
        
        # 2. 取得前端傳來的 JSON 資料
        data = request.get_json()
        st_slope = float(data['st_slope'])
        angina = float(data['angina'])
        
        # 3. 建立 Pandas DataFrame (必須和訓練時的欄位名稱/順序一致)
        feature_names = ['ST 段斜率', '運動是否誘發心絞痛']
        input_df = pd.DataFrame({
            'ST 段斜率': [st_slope],
            '運動是否誘發心絞痛': [angina]
        }, columns=feature_names)

        # 4. 執行預測
        # (A) 預測類別 (0 或 1)
        prediction_class = int(model_plot.predict(input_df)[0])
        
        # (B) 預測機率 (例如 [0.8, 0.2])
        #     我們只需要「有心臟病 (Class 1)」的機率
        probability_class_1 = model_plot.predict_proba(input_df)[0][1]

        return jsonify({
            "success": True,
            "prediction_class": prediction_class,
            "probability": round(probability_class_1, 4)
            # (我們回傳機率，但 JS 端會用 6 種固定文字)
        })

    except Exception as e:
        print(f"預測時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# -----------------------------------------------------------------
#     決策樹 API (保持不變)
# -----------------------------------------------------------------
def get_decision_tree_results():
    # ... (你現有的決策樹快取函式 - 不變) ...
    """
    【已修改】
    強制使用 'ST 段斜率' 和 '運動是否誘發心絞痛' 訓練模型，
    以匹配前端 UI 的互動預測功能。
    """
    # 檢查全域快取
    if "best_model" in dt_cache:
        # 如果快取中已有資料，直接回傳
        return dt_cache

    print("--- 正在訓練決策樹模型 (GridSearchCV)，請稍候... ---")
    
    # --- 執行一次您完整的 Colab 流程 ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'heart.csv')
    qust_cleaned = pd.read_csv(csv_path)

    target = '是否有心臟病'
    y = qust_cleaned[target]
    
    # 1. --- 【重點修改】 ---
    # 移除自動篩選
    # corr_matrix = qust_cleaned.corr()
    # ...
    # 改為「固定特徵」，以匹配 UI
    selected_features = ['ST 段斜率', '運動是否誘發心絞痛']
    
    # 檢查特徵是否存在
    for feature in selected_features:
        if feature not in qust_cleaned.columns:
            raise KeyError(f"資料集 'heart.csv' 中缺少必要的特徵: {feature}")
            
    X_selected = qust_cleaned[selected_features]
    # --- 【修改結束】 ---

    # 2. 資料分割 (不變)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # 3. 決策樹調參 (GridSearchCV) (不變)
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_dt_model = grid_search.best_estimator_

    # 4. 評估 (不變)
    y_pred = best_dt_model.predict(X_test)
    y_pred_proba = best_dt_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(report['weighted avg']['precision'], 4),
        "recall": round(report['weighted avg']['recall'], 4),
        "f1": round(report['weighted avg']['f1-score'], 4),
        "auc": round(auc_score, 4)
    }
    
    # 5. 產生 DOT data (不變)
    dot_data = export_graphviz(
        best_dt_model,
        out_file=None,
        feature_names=X_selected.columns,
        class_names=['無心臟病', '有心臟病'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    # 6. 儲存所有結果到快取中 (不變)
    dt_cache["best_model"] = best_dt_model
    dt_cache["metrics"] = metrics
    dt_cache["dot_data"] = dot_data
    dt_cache["description"] = {
        "dataset": "心臟衰竭資料集",
        "selected_features": selected_features, # 這現在會是 ['ST 段斜率', '運動是否誘發心絞痛']
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": len(qust_cleaned),
        "target": target,
        "best_params": grid_search.best_params_
    }
    
    print("--- 決策樹模型訓練完畢並已快取！ ---")
    return dt_cache


@app.route("/api/decision_tree/data")
def decision_tree_data():
    # ... (你現有的決策樹 /data API - 不變) ...
    """
    【已優化】決策樹 API - 僅提供「評估指標」
    """
    try:
        # 從快取中取得 results (如果快取為空, get_decision_tree_results 會自動訓練)
        results = get_decision_tree_results()
        
        # 直接回傳快取中的資料
        response = {
            "success": True,
            "metrics": results["metrics"],
            "description": results["description"]
        }
        return jsonify(response)
    except Exception as e:
        print(f"API 錯誤 (/data): {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/decision_tree/graph")
def decision_tree_graph():
    # ... (你現有的決策樹 /graph API - 不變) ...
    """
    【已優化】動態產生 Graphviz 樹狀結構圖 (SVG)
    """
    try:
        # 從快取中取得 results (如果快取為空, get_decision_tree_results 會自動訓練)
        results = get_decision_tree_results()
        
        # 從快取中取得 DOT data
        dot_data = results["dot_data"]
        
        # 6. 使用 Graphviz 動態產生 SVG
        graph = graphviz.Source(dot_data)
        svg_data = graph.pipe(format='svg')

        # 7. 回傳 SVG 圖片
        return Response(svg_data, mimetype='image/svg+xml')

    except Exception as e:
        print(f"產生 Graphviz 錯誤: {e}")
        return f"產生樹狀圖時發生錯誤: {e}", 500


@app.route("/api/decision_tree/predict", methods=['POST'])
def decision_tree_predict():
    # ... (你現有的決策樹 /predict API (v9) - 不變) ...
    """
    (v9 - 最終修正版)
    
    Bug 原因： v8 使用了 Numpy 陣列 (input_array) 餵給模型，
             但模型是用 Pandas DataFrame 訓練的，導致格式不符而崩潰。
             
    修正：   我們必須使用 Pandas DataFrame (input_df) 來餵食模型，
             這才是 model.apply() 和 model.decision_path() 
             唯一能 100% 正確辨識的格式。
    """
    try:
        # 1. 取得快取的模型
        results = get_decision_tree_results()
        model = results["best_model"]
        # 我們從快取中「知道」特徵順序
        feature_names = results["description"]["selected_features"]
        
        # 2. 取得前端傳來的 JSON 資料
        data = request.get_json()
        st_slope = float(data['st_slope'])
        angina = float(data['angina'])
        
        # 3. 【【【關鍵 Bug 修正】】】
        #    我們「必須」建立 Pandas DataFrame，
        #    因為模型是用這個格式訓練的。
        
        input_df = pd.DataFrame({
            'ST 段斜率': [st_slope],
            '運動是否誘發心絞痛': [angina]
        }, columns=feature_names) # 確保欄位順序正確

        # --- 【【【修正完畢】】】 ---

        # 4. 執行預測 (現在餵給它 input_df)
        
        # (A) 取得決策路徑 (用於高亮)
        #    (這個 v8 的 .indices.tolist() 語法是正確的)
        path_sparse_matrix = model.decision_path(input_df)
        path_nodes = path_sparse_matrix.indices.tolist()

        # (B) 取得最終葉節點 (Leaf ID)。
        #    (這行現在 100% 會成功)
        leaf_id = int(model.apply(input_df)[0])
        
        # (C) 取得葉節點上的 value
        values = model.tree_.value[leaf_id].flatten().tolist()
        
        # (D) 計算「有心臟病」(Class 1) 的機率
        total_samples = sum(values)
        prob_class_1 = 0.0
        if total_samples > 0:
            prob_class_1 = values[1] / total_samples

        return jsonify({
            "success": True,
            "path_nodes": path_nodes,   # <--- 這次 100% 正確
            "leaf_id": leaf_id,         # <--- 這次 100% 正確
            "values": values,
            "probability": round(prob_class_1, 4)
        })

    except Exception as e:
        print(f"預測時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# --- 啟動應用 (不變) ---
def main():
    """啟動應用（教學用：啟用 debug 模式）"""
    print("--- 伺服器已啟動，正在監聽 http://0.0.0.0:5000 ---")
    print("--- 警告：Debug 模式已開啟，請勿用於生產環境 ---")
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()