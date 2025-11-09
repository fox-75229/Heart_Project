from flask import Flask,render_template, jsonify
import numpy as np
# 載入 Pandas 讀取csv
import pandas as pd
import os # 確保檔案路徑正確

from sklearn.datasets import fetch_california_housing
# 載入邏輯迴歸模型
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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
            raise FileNotFoundError(f"找不到檔案:{csv_path}。請確認")
        #讀取'csv_path'
        data = pd.read_csv(csv_path)
        
        # 取得特徵
        feature_1 = 'ST_Slope'
        feature_2 = 'ExerciseAngina'
        target = 'HeartDisease'

        if feature_1 not in data.columns or feature_2 not in data.columns or target not in data.columns:
            raise KeyError(f"csv 檔案缺少'{feature_1}'、'{feature_2}'or'{target}'欄位'。")
        
        x = data[[feature_1, feature_2]]
        y = data[target]

        #分割資料
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y
        )
        # 訓練邏輯迴歸
        model = LogisticRegression(random_state=42, solver='liblinear')
        model.fit(X_train, y_train)

        #產生決策邊界
        x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
        y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
        #網格密度
        step = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

        #預測
        meshgrid_data = np.c_[xx.ravel(), yy.ravel()]
        meshgrid_data_df = pd.DataFrame(meshgrid_data, columns=[feature_1, feature_2])
        Z = model.predict(meshgrid_data_df)
        Z = Z.reshape(xx.shape)

        # 計算評估指標
        y_pred_test = model.predict(X_test)
        # AUC 
        y_pred_test = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred_test), 4),
            "precision": round(precision_score(y_test, y_pred_test), 4),
            "recall": round(recall_score(y_test, y_pred_test), 4),
            "f1": round(f1_score(y_test, y_pred_test), 4),
            "auc": round(roc_auc_score(y_test, y_pred_test), 4)
        }
        # 準備json回應資料
        response = {
            "success": True,
            "data":{

                #散點圖資料
                "train_points":{
                    "x1": X_train[feature_1].tolist(),
                    "x2": X_train[feature_2].tolist(),
                    "y": y_train.tolist()
                },
                "test_points":{
                    "x1": X_test[feature_1].tolist(),
                    "x2": X_test[feature_2].tolist(),
                    "y": y_test.tolist()
                },
                #決策邊界資料
                "decision_boundary":{
                    "xx": xx.tolist(),
                    "yy": yy.tolist(),
                    "Z": Z.tolist()
                }
            },
            "metrics": metrics,
            "description":{
                "dataset": "心臟衰竭資料集",
                "x1_feature": feature_1,
                "x2_feature": feature_2,
                "y_target": target,
                "info": "圖表顯示邏輯迴歸模型如何使用 2 個特徵來劃分決策邊界。"
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