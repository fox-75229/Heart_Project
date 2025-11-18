"""
主應用程式檔案 (app.py)

一個 Flask 伺服器，提供機器學習模型的 API 和網頁介面。
包含兩個主要模型：
1. 邏輯迴歸 (Logistic Regression) - 用於分類和視覺化決策邊界。
2. 決策樹 (Decision Tree) - 用於分類和視覺化決策過程。

使用全域快取 (cache) 來儲存訓練好的模型，避免重複運算。
"""

# --- 1. 匯入套件 ---

# Flask 框架相關
from flask import Flask, render_template, jsonify, Response, request
import os

# 資料處理
import pandas as pd
import numpy as np

# 機器學習 (Scikit-learn)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)

# 視覺化
import graphviz

# --- 2. 應用程式設定 (Flask) ---

app = Flask(__name__)
app.json.ensure_ascii = False  # 讓 API 的 JSON 回應支援中文

# --- 3. 全域快取 ---
# 這些字典用來快取訓練好的模型和結果，避免每次 API 請求都重新訓練
dt_cache = {}  # 決策樹模型快取
lr_cache = {}  # 邏輯迴歸模型快取

# --- 4. 網頁路由 (Pages) ---

@app.route("/")
def index():
    """首頁"""
    return render_template("index.html")

@app.route("/decision_tree")
def decision_tree():
    """決策樹展示頁面"""
    return render_template("decision_tree.html")

@app.route("/logistic")
def logistic():
    """邏輯迴歸展示頁面"""
    return render_template("logistic.html")

# -----------------------------------------------------------------
# 5. 邏輯迴歸 (Logistic Regression) - 核心邏輯與 API
# -----------------------------------------------------------------

def get_logistic_results():
    """
    訓練邏輯迴歸模型並快取結果。

    如果快取中已有結果，則直接回傳快取。
    此函式會執行兩個訓練任務：
    1.  **高分模型 (model_best)**: 
        使用 *所有特徵* 進行訓練，以獲得最佳的評估指標 (metrics)。
    2.  **圖表模型 (model_plot)**: 
        僅使用 *兩個特徵* ('ST 段斜率', '運動是否誘發心絞痛') 進行訓練，
        主要用於前端的 2D 視覺化和該圖表上的即時預測。
    
    Returns:
        dict: 包含模型、指標和圖表資料的快取字典 (lr_cache)。
    """
    # 檢查快取，如果 'model_plot' 已存在，表示已訓練過
    if "model_plot" in lr_cache:
        return lr_cache
    
    print("--- (快取未命中) 正在訓練邏輯迴歸模型 ---")

    # 1. 讀取資料
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'heart.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到資料檔案: {csv_path}")
    qust_cleaned = pd.read_csv(csv_path)
    
    target = '是否有心臟病'

    # 2. 訓練「高分模型」（使用所有特徵）以取得最佳評估指標
    X_all = qust_cleaned.drop(columns=[target])
    y_all = qust_cleaned[target]
    
    # 使用 stratify=y_all 確保訓練集和測試集中的類別比例相同
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    
    model_best = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
    model_best.fit(X_train_all, y_train_all)
    
    # 預測並計算評估指標
    y_pred_class_best = model_best.predict(X_test_all)
    y_pred_proba_best = model_best.predict_proba(X_test_all)[:, 1] # 取出類別 1 的機率

    # 儲存最佳模型的評估指標
    best_metrics = {
        "accuracy": round(accuracy_score(y_test_all, y_pred_class_best), 4),
        "precision": round(precision_score(y_test_all, y_pred_class_best), 4),
        "recall": round(recall_score(y_test_all, y_pred_class_best), 4),
        "f1": round(f1_score(y_test_all, y_pred_class_best), 4),
        "auc": round(roc_auc_score(y_test_all, y_pred_proba_best), 4)
    }
    
    # 3. 訓練「圖表模型」（僅使用兩個特徵）
    feature_1 = 'ST 段斜率'
    feature_2 = '運動是否誘發心絞痛'
    x_plot = qust_cleaned[[feature_1, feature_2]]
    y_plot = qust_cleaned[target]
    
    X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
        x_plot, y_plot, test_size=0.3, random_state=42, stratify=y_plot
    )
    
    model_plot = LogisticRegression(random_state=42, solver='liblinear')
    model_plot.fit(X_train_plot, y_train_plot)
    
    # 產生決策邊界資料 (Meshgrid)
    # 找出兩個特徵的
    x_min, x_max = x_plot.iloc[:, 0].min() - 0.5, x_plot.iloc[:, 0].max() + 0.5
    y_min, y_max = x_plot.iloc[:, 1].min() - 0.5, x_plot.iloc[:, 1].max() + 0.5
    
    # 建立網格點
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # 將網格點轉換為 DataFrame 格式，以便模型預測
    meshgrid_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[feature_1, feature_2])
    
    # 預測網格中每個點的類別
    Z = model_plot.predict(meshgrid_data).reshape(xx.shape)

    # 4. 組合圖表所需的資料
    chart_data = {
        # 訓練集點位
        "train_points": { "x1": X_train_plot[feature_1].tolist(), "x2": X_train_plot[feature_2].tolist(), "y": y_train_plot.tolist() },
        # 測試集點位
        "test_points": { "x1": X_test_plot[feature_1].tolist(), "x2": X_test_plot[feature_2].tolist(), "y": y_test_plot.tolist() },
        # 決策邊界
        "decision_boundary": { "xx": xx.tolist(), "yy": yy.tolist(), "Z": Z.tolist() }
    }
    
    # 組合圖表描述資訊
    chart_description = {
        "dataset": "心臟衰竭資料集",
        "x1_feature": feature_1,
        "x2_feature": feature_2,
        "target": target,
        "total_samples": len(qust_cleaned),
        "train_size": len(X_train_all), # 注意：這裡使用 all-feature 模型的樣本數
        "test_size": len(X_test_all),
        "selected_features": [feature_1, feature_2],
        "y_min": y_min # 用於圖表 Y 軸的最小邊界
    }

    # 5. 儲存結果到全域快取
    lr_cache["model_plot"] = model_plot           # 用於 2D 預測的模型
    lr_cache["best_metrics"] = best_metrics       # 使用所有特徵的最佳指標
    lr_cache["chart_data"] = chart_data           # 2D 圖表資料
    lr_cache["chart_description"] = chart_description # 圖表描述
    
    print("--- 邏輯迴歸模型訓練完成並已快取 ---")
    return lr_cache


@app.route("/api/logistic/data")
def logistic_data():
    """
    API 端點：取得邏輯迴歸的圖表資料和評估指標。
    
    Returns:
        JSON: 包含圖表資料、最佳模型指標和描述的 JSON 物件。
    """
    try:
        # 呼叫核心函式，如果已快取會立即回傳
        results = get_logistic_results()
        
        response = {
            "success": True,
            "data": results["chart_data"],
            "metrics": results["best_metrics"],
            "description": results["chart_description"]
        }
        return jsonify(response)
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except KeyError as e:
        # 如果 CSV 缺少 '是否有心臟病' 或其他關鍵欄位
        return jsonify({"success": False, "error": f"CSV 欄位錯誤: {e}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"伺服器錯誤: {e}"}), 500


@app.route("/api/logistic/predict", methods=['POST'])
def logistic_predict():
    """
    API 端點：使用「圖表模型」（兩個特徵）預測新資料。
    
    Input JSON:
        { "st_slope": float, "angina": float }
    
    Returns:
        JSON: 包含預測類別和機率的 JSON 物件。
    """
    try:
        # 取得快取的模型
        results = get_logistic_results()
        model_plot = results["model_plot"]
        
        data = request.get_json()
        st_slope = float(data['st_slope'])
        angina = float(data['angina'])
        
        # 準備模型所需的 DataFrame 格式
        input_df = pd.DataFrame({
            'ST 段斜率': [st_slope],
            '運動是否誘發心絞痛': [angina]
        })

        # 進行預測
        prediction_class = int(model_plot.predict(input_df)[0])
        probability_class_1 = model_plot.predict_proba(input_df)[0][1] # 類別 1 (有心臟病) 的機率

        return jsonify({
            "success": True,
            "prediction_class": prediction_class,
            "probability": round(probability_class_1, 4)
        })
    except Exception as e:
        import traceback
        traceback.print_exc() # 在伺服器控制台印出詳細錯誤
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------------------------------------------------
# 6. 決策樹 (Decision Tree) - 核心邏輯與 API
# -----------------------------------------------------------------
def get_decision_tree_results():
    """
    訓練決策樹模型並快取結果。

    如果快取中已有結果，則直接回傳快取。
    此函式會：
    1.  讀取資料。
    2.  僅使用 'ST 段斜率' 和 '運動是否誘發心絞痛' 兩個特徵。
    3.  使用 `GridSearchCV` 進行交叉驗證，自動找出最佳超參數（如 max_depth）。
    4.  計算最佳模型的評估指標。
    5.  產生用於視覺化的 `dot_data` (Graphviz 格式)。
    
    Returns:
        dict: 包含模型、指標和圖表資料的快取字典 (dt_cache)。
    """
    # 檢查快取
    if "best_model" in dt_cache:
        return dt_cache

    print("--- (快取未命中) 正在訓練決策樹模型 ---")
    
    # 1. 讀取資料
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'heart.csv')
    qust_cleaned = pd.read_csv(csv_path)
    
    target = '是否有心臟病'
    y = qust_cleaned[target]

    # 2. 特徵工程：僅使用固定的兩個特徵
    selected_features = ['ST 段斜率', '運動是否誘發心絞痛']
    X_selected = qust_cleaned[selected_features]

    # 3. 資料分割
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # 4. 模型訓練：使用 GridSearchCV 尋找最佳參數
    # 定義要嘗試的參數網格
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],      # 樹的最大深度
        'min_samples_split': [2, 5, 10],   # 節點再分割所需的最小樣本數
        'min_samples_leaf': [1, 2, 4]      # 葉節點最少的樣本數
    }
    
    dt_model = DecisionTreeClassifier(random_state=42)
    
    # 使用 5 折交叉驗證 (cv=5) 和 'accuracy' 作為評分標準
    grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 取得GridSearch找到的最佳模型
    best_dt_model = grid_search.best_estimator_

    # 5. 模型評估
    y_pred = best_dt_model.predict(X_test)
    y_pred_proba = best_dt_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # 儲存評估指標
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(report['weighted avg']['precision'], 4),
        "recall": round(report['weighted avg']['recall'], 4),
        "f1": round(report['weighted avg']['f1-score'], 4),
        "auc": round(auc_score, 4)
    }
    
    # 6. 產生 Graphviz 視覺化資料 (DOT 格式)
    dot_data = export_graphviz(
        best_dt_model,
        out_file=None,
        feature_names=X_selected.columns,
        class_names=['無心臟病', '有心臟病'], # 類別 0 和 1 的名稱
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    # 7. 儲存到快取
    dt_cache["best_model"] = best_dt_model
    dt_cache["metrics"] = metrics
    dt_cache["dot_data"] = dot_data
    dt_cache["description"] = {
        "dataset": "心臟衰竭資料集",
        "selected_features": selected_features,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": len(qust_cleaned),
        "target": target,
        "best_params": grid_search.best_params_ # 儲存 GridSearch 找到的最佳參數
    }
    
    print(f"--- 決策樹模型訓練完成 (最佳參數: {grid_search.best_params_}) ---")
    return dt_cache


@app.route("/api/decision_tree/data")
def decision_tree_data():
    """
    API 端點：取得決策樹模型的評估指標和描述資訊。
    
    Returns:
        JSON: 包含 metrics 和 description 的 JSON 物件。
    """
    try:
        results = get_decision_tree_results()
        response = {
            "success": True,
            "metrics": results["metrics"],
            "description": results["description"]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/decision_tree/graph")
def decision_tree_graph():
    """
    API 端點：產生並回傳決策樹的 SVG 影像。
    
    Returns:
        Response: Mimetype 為 'image/svg+xml' 的 SVG 影像。
    """
    try:
        results = get_decision_tree_results()
        
        # 使用 graphviz 將 DOT 字串轉換為 SVG
        graph = graphviz.Source(results["dot_data"])
        svg_data = graph.pipe(format='svg')
        
        return Response(svg_data, mimetype='image/svg+xml')
    except Exception as e:
        return f"產生樹狀圖錯誤: {e}", 500


@app.route("/api/decision_tree/predict", methods=['POST'])
def decision_tree_predict():
    """
    API 端點：預測單筆資料，並回傳其在決策樹中的「路徑」和「葉節點」資訊。
    這對於在前端 SVG 上視覺化決策過程非常有用。
    
    Input JSON:
        { "st_slope": float, "angina": float }
        
    Returns:
        JSON: 包含決策路徑 (path_nodes)、葉節點 ID (leaf_id) 和機率 (probability) 的物件。
    """
    try:
        results = get_decision_tree_results()
        model = results["best_model"]
        feature_names = results["description"]["selected_features"]

        # 1. 取得輸入資料
        data = request.get_json()
        st_slope = float(data['st_slope'])
        angina = float(data['angina'])

        # 2. 準備模型所需的 DataFrame
        # 確保 columns 順序與訓練時一致
        input_df = pd.DataFrame({
            'ST 段斜率': [st_slope],
            '運動是否誘發心絞痛': [angina]
        }, columns=feature_names)

        # 3. 取得決策路徑和葉節點
        # .decision_path() 回傳一個稀疏矩陣，.indices 包含路徑上的所有節點 ID
        path_nodes = model.decision_path(input_df).indices.tolist()
        
        # .apply() 回傳每個樣本最終落入的葉節點 ID
        leaf_id = int(model.apply(input_df)[0])
        
        # 4. 從模型的 .tree_ 屬性中提取葉節點的值 (樣本分佈)
        # model.tree_.value[leaf_id] 會回傳 [[class_0_count, class_1_count]]
        values = model.tree_.value[leaf_id].flatten().tolist()
        total_samples = sum(values)
        
        # 計算類別 1 (有心臟病) 的機率
        prob_class_1 = values[1] / total_samples if total_samples > 0 else 0.0

        return jsonify({
            "success": True,
            "path_nodes": path_nodes,  # 決策路徑上的所有節點 ID
            "leaf_id": leaf_id,      # 最終的葉節點 ID
            "values": values,      # 葉節點的樣本分佈 [無心臟病, 有心臟病]
            "probability": round(prob_class_1, 4) # 預測為「有心臟病」的機率
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# --- 7. 啟動應用 ---

def main():
    """
    主函式：啟動 Flask 伺服器。
    """
    print("--- 伺服器啟動，監聽 http://0.0.0.0:5000 ---")
    print("--- Debug 模式已開啟，請勿用於生產環境 ---")
    # debug=True 會在程式碼變更時自動重啟伺服器
    # host='0.0.0.0' 讓區域網路中的其他裝置可以存取
    app.run(debug=True) #個人開發測試用
    # app.run(debug=True, host='0.0.0.0', port=5000) 
    #報告其他設備演示用，誤用於部屬

if __name__ == "__main__":
    main()