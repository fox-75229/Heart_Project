from flask import Flask, render_template, jsonify, Response # 【新】匯入 Response
import pandas as pd
import numpy as np
import os
import io # 【新】用於處理圖片 I/O

# 【新】匯入決策樹和 Graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz # 【新】匯入 Graphviz

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#---Flask 應用設定---
app = Flask(__name__)

# 自定義JSON序列化設定
app.json.ensure_ascii = False


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/knn")
def knn():
    return render_template("knn.html")

@app.route("/decision_tree")
def decision_tree():
    return render_template("decision_tree.html")

@app.route("/logistic")
def logistic():
    return render_template("logistic.html")

#---API 路由---

@app.route("/api/logistic/data")
def logistic_data():
    """邏輯迴歸 API - 使用心臟衰竭資料集"""
    try:
        # 載入心臟衰竭資料集heart.csv
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'data', 'heart.csv')

        # 檢查檔案是否正確載入
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到檔案:{csv_path}")
        
        #讀取'csv_path'
        qust_cleaned = pd.read_csv(csv_path)
        
        # -----------------------------------------------------------------
        # 2. [A 部分] 訓練「高分模型」 (All Features)
        # -----------------------------------------------------------------
        
        target = '是否有心臟病'
        if target not in qust_cleaned.columns:
            raise KeyError(f"CSV 檔案缺少目標欄位: '{target}'")
            
        X_all = qust_cleaned.drop(columns=[target])
        y_all = qust_cleaned[target]
        
        # 依照您 Colab 的 test_size=0.3
        X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
            X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
        )
        
        # 訓練「高分模型」 (不縮放, 匹配 Colab)
        model_best = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
        model_best.fit(X_train_all, y_train_all)

        # 計算「高分」指標
        y_pred_class_best = model_best.predict(X_test_all)
        y_pred_proba_best = model_best.predict_proba(X_test_all)[:, 1]

        best_metrics = {
            "accuracy": round(accuracy_score(y_test_all, y_pred_class_best), 4),
            "precision": round(precision_score(y_test_all, y_pred_class_best), 4),
            "recall": round(recall_score(y_test_all, y_pred_class_best), 4),
            "f1": round(f1_score(y_test_all, y_pred_class_best), 4),
            "auc": round(roc_auc_score(y_test_all, y_pred_proba_best), 4)
        }
        # (這組數字現在會 100% 即時計算，並接近 0.8462)

        # -----------------------------------------------------------------
        # 3. [B 部分] 訓練「圖表模型」 (2 Features)
        # -----------------------------------------------------------------
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

        # 產生決策邊界
        x_min, x_max = x_plot.iloc[:, 0].min() - 0.5, x_plot.iloc[:, 0].max() + 0.5
        y_min, y_max = x_plot.iloc[:, 1].min() - 0.5, x_plot.iloc[:, 1].max() + 0.5
        step = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

        meshgrid_data = np.c_[xx.ravel(), yy.ravel()]
        meshgrid_data_df = pd.DataFrame(meshgrid_data, columns=[feature_1, feature_2])
        Z = model_plot.predict(meshgrid_data_df)
        Z = Z.reshape(xx.shape)
        
        # --- 4. [C 部分] 合併回應 ---
        response = {
            "success": True,
            "data":{ # 來自「B 部分」的圖表資料
                "train_points":{
                    "x1": X_train_plot[feature_1].tolist(),
                    "x2": X_train_plot[feature_2].tolist(),
                    "y": y_train_plot.tolist()
                },
                "test_points":{
                    "x1": X_test_plot[feature_1].tolist(),
                    "x2": X_test_plot[feature_2].tolist(),
                    "y": y_test_plot.tolist()
                },
                "decision_boundary":{
                    "xx": xx.tolist(),
                    "yy": yy.tolist(),
                    "Z": Z.tolist()
                }
            },
            "metrics": best_metrics, # 來自「A 部分」的高分指標
            "description":{
                "dataset": "心臟衰竭預測資料集 (已清理)",
                "x1_feature": feature_1,
                "x2_feature": feature_2,
                "y_target": target,
                "info": "圖表顯示2D模型，指標顯示All-Feature模型。"
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
#     決策樹 API
# -----------------------------------------------------------------

@app.route("/api/decision_tree/graph")
def decision_tree_graph():
    """
    動態產生 Graphviz 樹狀結構圖 (SVG)
    決策樹 API - 使用心臟衰竭資料集
    """
    try:
        # 1. 載入資料
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'data', 'heart.csv')
        qust_cleaned = pd.read_csv(csv_path)

        target = '是否有心臟病'
        feature_1 = 'ST 段斜率'
        feature_2 = '運動是否誘發心絞痛'
        
        # 2. 100% 複製您的 Colab 流程
        # (您的 Colab 決策樹只用了這 2 個特徵)
        X_selected = qust_cleaned[[feature_1, feature_2]]
        y = qust_cleaned[target]
        
        # 3. 分割 (使用您 Colab 的 test_size=0.3)
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42
        )
        
        # 4. 訓練 (使用您 Colab 的模型: DecisionTreeClassifier(random_state=42))
        decision_tree_model = DecisionTreeClassifier(random_state=42)
        decision_tree_model.fit(X_train, y_train) # 在 X_train 上訓練

        # 5. 產生 Graphviz DOT data
        dot_data = export_graphviz(
            decision_tree_model, # 使用這個模型
            out_file=None,
            feature_names=X_selected.columns, # ['ST 段斜率', '運動是否誘發心絞痛']
            class_names=['無心臟病', '有心臟病'], # 您的 Colab 設定
            filled=True,
            rounded=True,
            special_characters=True
        )

        # 6. 使用 Graphviz 動態產生 SVG (在記憶體中)
        graph = graphviz.Source(dot_data)
        svg_data = graph.pipe(format='svg')

        # 7. 回傳 SVG 圖片
        return Response(svg_data, mimetype='image/svg+xml')

    except Exception as e:
        print(f"產生 Graphviz 錯誤: {e}")
        # (檢查您的終端機是否有 'ExecutableNotFound: failed to execute "dot"')
        return f"產生樹狀圖時發生錯誤: {e}", 500


def main():
    """啟動應用（教學用：啟用 debug 模式）"""
    # 在開發環境下使用 debug=True，部署時請關閉
    app.run(debug=True)

if __name__ == "__main__":
    main()