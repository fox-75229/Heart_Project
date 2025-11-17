from flask import Flask, render_template, jsonify, Response, request
import pandas as pd
import numpy as np
import os
import io  # 處理圖片 I/O

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# --- Flask 應用設定 ---
app = Flask(__name__)
app.json.ensure_ascii = False  # JSON 支援中文

# --- 全域快取 ---
dt_cache = {}  # 決策樹快取
lr_cache = {}  # 邏輯迴歸快取

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

# -----------------------------------------------------------------
# 邏輯迴歸模型相關
# -----------------------------------------------------------------
def get_logistic_results():
    """
    訓練邏輯迴歸模型並儲存到快取。
    包含：
      - 使用所有特徵訓練高分模型
      - 使用兩個特徵訓練圖表模型
      - 計算評估指標並準備可視化資料
    """
    # 檢查快取
    if "model_plot" in lr_cache:
        return lr_cache
    
    print("--- 訓練邏輯迴歸模型 ---")

    # 1. 讀取資料
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'heart.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到檔案: {csv_path}")
    qust_cleaned = pd.read_csv(csv_path)
    
    # 2. 訓練使用所有特徵的模型
    target = '是否有心臟病'
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
    
    # 3. 訓練圖表模型（兩個特徵）
    feature_1 = 'ST 段斜率'
    feature_2 = '運動是否誘發心絞痛'
    x_plot = qust_cleaned[[feature_1, feature_2]]
    y_plot = qust_cleaned[target]
    
    X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
        x_plot, y_plot, test_size=0.3, random_state=42, stratify=y_plot
    )
    
    model_plot = LogisticRegression(random_state=42, solver='liblinear')
    model_plot.fit(X_train_plot, y_train_plot)
    
    # 產生決策邊界資料
    x_min, x_max = x_plot.iloc[:, 0].min() - 0.5, x_plot.iloc[:, 0].max() + 0.5
    y_min, y_max = x_plot.iloc[:, 1].min() - 0.5, x_plot.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    meshgrid_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[feature_1, feature_2])
    Z = model_plot.predict(meshgrid_data).reshape(xx.shape)

    # 4. 組合圖表資料
    chart_data = {
        "train_points": { "x1": X_train_plot[feature_1].tolist(), "x2": X_train_plot[feature_2].tolist(), "y": y_train_plot.tolist() },
        "test_points": { "x1": X_test_plot[feature_1].tolist(), "x2": X_test_plot[feature_2].tolist(), "y": y_test_plot.tolist() },
        "decision_boundary": { "xx": xx.tolist(), "yy": yy.tolist(), "Z": Z.tolist() }
    }
    
    chart_description = {
        "dataset": "心臟衰竭資料集",
        "x1_feature": feature_1,
        "x2_feature": feature_2,
        "total_samples": len(qust_cleaned),
        "train_size": len(X_train_all),
        "test_size": len(X_test_all),
        "target": target,
        "selected_features": [feature_1, feature_2],
        "y_min": y_min
    }

    # 5. 儲存到快取
    lr_cache["model_plot"] = model_plot
    lr_cache["best_metrics"] = best_metrics
    lr_cache["chart_data"] = chart_data
    lr_cache["chart_description"] = chart_description
    
    print("--- 邏輯迴歸模型訓練完成 ---")
    return lr_cache


@app.route("/api/logistic/data")
def logistic_data():
    """
    回傳邏輯迴歸圖表資料與評估指標
    """
    try:
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
        return jsonify({"success": False, "error": f"CSV 欄位錯誤: {e}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"伺服器錯誤: {e}"}), 500


@app.route("/api/logistic/predict", methods=['POST'])
def logistic_predict():
    """
    使用圖表模型預測新資料
    """
    try:
        results = get_logistic_results()
        model_plot = results["model_plot"]
        
        data = request.get_json()
        st_slope = float(data['st_slope'])
        angina = float(data['angina'])
        
        input_df = pd.DataFrame({
            'ST 段斜率': [st_slope],
            '運動是否誘發心絞痛': [angina]
        })

        prediction_class = int(model_plot.predict(input_df)[0])
        probability_class_1 = model_plot.predict_proba(input_df)[0][1]

        return jsonify({
            "success": True,
            "prediction_class": prediction_class,
            "probability": round(probability_class_1, 4)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------------------------------------------------
# 決策樹模型相關
# -----------------------------------------------------------------
def get_decision_tree_results():
    """
    訓練決策樹模型並儲存到快取
    """
    if "best_model" in dt_cache:
        return dt_cache

    print("--- 訓練決策樹模型 ---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'heart.csv')
    qust_cleaned = pd.read_csv(csv_path)
    
    target = '是否有心臟病'
    y = qust_cleaned[target]

    # 使用固定特徵
    selected_features = ['ST 段斜率', '運動是否誘發心絞痛']
    X_selected = qust_cleaned[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_dt_model = grid_search.best_estimator_

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
    
    dot_data = export_graphviz(
        best_dt_model,
        out_file=None,
        feature_names=X_selected.columns,
        class_names=['無心臟病', '有心臟病'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    
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
        "best_params": grid_search.best_params_
    }
    
    print("--- 決策樹模型訓練完成 ---")
    return dt_cache


@app.route("/api/decision_tree/data")
def decision_tree_data():
    """回傳決策樹評估指標"""
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
    """產生決策樹 SVG"""
    try:
        results = get_decision_tree_results()
        graph = graphviz.Source(results["dot_data"])
        svg_data = graph.pipe(format='svg')
        return Response(svg_data, mimetype='image/svg+xml')
    except Exception as e:
        return f"產生樹狀圖錯誤: {e}", 500


@app.route("/api/decision_tree/predict", methods=['POST'])
def decision_tree_predict():
    """預測單筆資料並回傳路徑、葉節點、機率"""
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

        path_nodes = model.decision_path(input_df).indices.tolist()
        leaf_id = int(model.apply(input_df)[0])
        values = model.tree_.value[leaf_id].flatten().tolist()
        total_samples = sum(values)
        prob_class_1 = values[1] / total_samples if total_samples > 0 else 0.0

        return jsonify({
            "success": True,
            "path_nodes": path_nodes,
            "leaf_id": leaf_id,
            "values": values,
            "probability": round(prob_class_1, 4)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# --- 啟動應用 ---
def main():
    print("--- 伺服器啟動，監聽 http://0.0.0.0:5000 ---")
    print("--- Debug 模式已開啟，勿用於生產環境 ---")
    app.run(debug=True)



if __name__ == "__main__":
    main()