from flask import Flask,render_template, jsonify
import pandas as pd# 載入 Pandas 讀取csv
import numpy as np
import os # 確保檔案路徑正確


from sklearn.datasets import fetch_california_housing
# 載入邏輯迴歸模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression

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

@app.route("/api/regression/data")
def regression_data():
    """線性迴歸 API - 使用加州房價資料集"""
    try:  

        # 載入加州房價資料集
        housing = fetch_california_housing()

        # 只使用前200筆資料作為展示
        sample_size = 200
        X_full = housing.data[:sample_size]
        y_full = housing.target[:sample_size] #房價(單位:十萬美元)

        # 使用**平均房間數**作為預測特徵(索引2)
        feature_idx = 2
        X = X_full[:,feature_idx].reshape(-1,1)
        y = y_full * 10

        # 分割訓練和測試資料(80/20)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        # 訓練線性迴歸模型
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 預測
        # 訓練資料的預測
        y_train_pred = model.predict(X_train)

        #測試資料的預測
        y_test_pred = model.predict(X_test)

        # 計算評估指標
        r2 = r2_score(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)

        # 生成迴歸線資料(用於繪圖)
        X_line = np.linspace(X.min(),X.max(), 100).reshape(-1,1)
        y_line = model.predict(X_line)

        # 準備回應資料
        response = {
            "success": True,
            "data":{
                "train":{
                    "x": X_train.flatten().tolist(),
                    "y": y_train.tolist(),
                    "y_pred": y_train_pred.tolist()
                },
                "test":{
                    "x": X_test.flatten().tolist(),
                    "y": y_test.tolist(),
                    "y_pred": y_test_pred.tolist()
                },
                "regression_line":{
                    "x": X_line.flatten().tolist(),
                    "y": y_line.tolist()
                }        
            },
            "metrics":{
                "r2_score": round(r2, 4),
                "mse": round(mse, 2),
                "rmse": round(rmse, 2),
                "coefficient": round(model.coef_[0], 2),
                "intercept": round(model.intercept_, 2)
            },
            "description":{
                "dataset":"加州房價資料集",
                "samples": len(y),
                "train_size": len(y_train),
                "test_size": len(y_test),
                "feature_name": "平均房間數",
                "feature_unit": "間",
                "target_name": "房價",
                "target_unit": "萬美元",
                "info": "此資料集取自 1990 年加州人口普查資料"
            }
        }

        return jsonify(response)
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": str(e)  
            }, 500
        )

@app.route("/api/regression/predict")
def regression_predict():
    """線性迴歸預測 API - 根據房間數預測房價"""
    response = {
        "success": True,
        "Prediction":{
            "price": 100,
            "unit": "萬美元"
        }
    }
    return jsonify(response)

def main():
    """啟動應用（教學用：啟用 debug 模式）"""
    # 在開發環境下使用 debug=True，部署時請關閉
    app.run(debug=True)

if __name__ == "__main__":
    main()