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

# --- 【新】建立一個全域快取 (Cache) ---
# 我們將在這裡儲存訓練好的決策樹模型和指標
dt_cache = {}

# --- 網頁路由 ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/decision_tree")
def decision_tree():
    return render_template("decision_tree.html")

@app.route("/logistic")
def logistic():
    return render_template("logistic.html")

#---API 路由---

@app.route("/api/logistic/data")
def logistic_data():
    """邏輯迴歸 API (不變)"""
    try:
        # (您的邏輯迴歸程式碼... 為了版面整潔，此處省略)
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
        # 4. [C 部分] 合併回應
        response = {
            "success": True,
            "data":{ 
                "train_points":{ "x1": X_train_plot[feature_1].tolist(), "x2": X_train_plot[feature_2].tolist(), "y": y_train_plot.tolist() },
                "test_points":{ "x1": X_test_plot[feature_1].tolist(), "x2": X_test_plot[feature_2].tolist(), "y": y_test_plot.tolist() },
                "decision_boundary":{ "xx": xx.tolist(), "yy": yy.tolist(), "Z": Z.tolist() }
            },
            "metrics": best_metrics,
            "description":{
                "dataset": "心臟衰竭資料集",
                "x1_feature": feature_1, # 用於 Plotly 標題
                "x2_feature": feature_2, # 用於 Plotly 標題
                "total_samples": len(qust_cleaned),
                "train_size": len(X_train_plot), # 使用「高分模型」的訓練集大小
                "test_size": len(X_test_plot),   # 使用「高分模型」的測試集大小
                "target": target,
                "selected_features": [feature_1, feature_2] 
            }
        }
        return jsonify(response)
    # 錯誤處理
    except FileNotFoundError as e:
        print(f"檔案錯誤: {e}")
        return jsonify({"success": False, "error": str(e)}), 404
    except KeyError as e:
        print(f"欄位錯誤: {e}")
        return jsonify({"success": False, "error": f"CSV 欄位錯誤: {e}"}), 400
    except Exception as e:
        print(f"伺服器錯誤: {e}") 
        return jsonify({"success": False, "error": f"伺服器內部錯誤: {e}"}), 500
    

# -----------------------------------------------------------------
#     決策樹 API 
# -----------------------------------------------------------------

def get_decision_tree_results():
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
        'max_depth': [2],
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

def get_leaf_id_to_svg_node_id(model):
    """
    建立 leaf_id 到 SVG node id 的對照表
    """
    # 遍歷所有節點，找到所有葉節點
    tree = model.tree_
    leaf_id_to_svg = {}
    node_count = tree.node_count
    for node_id in range(node_count):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        # 葉節點: 左右都是 -1
        if left == -1 and right == -1:
            leaf_id_to_svg[node_id] = f"node{node_id}"
    return leaf_id_to_svg

@app.route("/api/decision_tree/predict", methods=['POST'])
def decision_tree_predict():
    try:
        results = get_decision_tree_results()
        model = results["best_model"]
        feature_names = results["description"]["selected_features"]

        data = request.get_json()
        st_slope = float(data['st_slope'])
        angina = float(data['angina'])

        input_df = pd.DataFrame({
            'ST 段斜率': [st_slope],
            '運動是否誘發心絞痛': [angina]
        }, columns=feature_names)

        path_sparse_matrix = model.decision_path(input_df)
        path_nodes = path_sparse_matrix.indices.tolist()
        leaf_id = int(model.apply(input_df)[0])

        # 建立 leaf_id 到 SVG node id 的對照表
        leaf_id_to_svg = get_leaf_id_to_svg_node_id(model)
        svg_node_id = leaf_id_to_svg.get(leaf_id, f"node{leaf_id}")

        values = model.tree_.value[leaf_id].flatten().tolist()
        total_samples = sum(values)
        prob_class_1 = 0.0
        if total_samples > 0:
            prob_class_1 = values[1] / total_samples

        return jsonify({
            "success": True,
            "path_nodes": path_nodes,
            "leaf_id": leaf_id,
            "svg_node_id": svg_node_id,  # 新增這個欄位
            "values": values,
            "probability": round(prob_class_1, 4)
        })

    except Exception as e:
        print(f"预测时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500



# --- 啟動應用 ---
def main():
    """啟動應用（教學用：啟用 debug 模式）"""
    print("--- 伺服器已啟動，正在監聽 http://0.0.0.0:5000 ---")
    print("--- 警告：Debug 模式已開啟，請勿用於生產環境 ---")
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()




