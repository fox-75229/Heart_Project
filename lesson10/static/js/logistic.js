// 1. 等待 DOM 載入
document.addEventListener("DOMContentLoaded", () => {
    // 呼叫主函式
    loadLogisticData();
});

/**
 * 2. 呼叫 API 並處理回應
 * (仿造您的 loadRegressionData 函式)
 */
async function loadLogisticData() {
    // 選取圖表容器和讀取提示
    const chartDiv = document.getElementById('plotly-chart');
    const loadingSpinner = chartDiv.querySelector('.loading-spinner');
    const loadingText = chartDiv.querySelector('.loading-text');

    try {
        // 發送 Fetch 請求到您的 Flask API
        const response = await fetch('/api/logistic/data');

        // 檢查網路回應是否正常 (e.g., 404, 500)
        if (!response.ok) {
            throw new Error(`HTTP 錯誤! 狀態: ${response.status}`);
        }

        const result = await response.json();

        // 檢查 API 回傳的 JSON 是否成功
        if (result.success) {
            // 成功：
            // 3. 更新評估指標
            updateMetrics(result.metrics);

            // 4. 繪製圖表
            drawPlot(result.data, result.description);

            // 繪製完成，移除讀取提示
            if (loadingSpinner) loadingSpinner.remove();
            if (loadingText) loadingText.remove();

        } else {
            // API 返回了 success: false
            throw new Error(result.error);
        }

    } catch (error) {
        // 5. 處理所有錯誤
        console.error("載入邏輯迴歸資料失敗:", error);
        // (仿造您的 showError 邏輯，但顯示在頁面上)
        chartDiv.innerHTML = `<p class="error-text">❌ 載入圖表失敗: ${error.message}</p>`;
    }
}

/**
 * 3. 更新評估指標
 * (仿造您的 updateMetrics 函式，但使用新的 ID)
 * @param {object} metrics - 從 API 傳來的 metrics 物件
 */
function updateMetrics(metrics) {
    // 根據您 HTML 中的 ID 更新文字
    document.getElementById('metrics-accuracy').textContent = metrics.accuracy;
    document.getElementById('metrics-precision').textContent = metrics.precision;
    document.getElementById('metrics-recall').textContent = metrics.recall;
    document.getElementById('metrics-f1').textContent = metrics.f1;
    document.getElementById('metrics-auc').textContent = metrics.auc;
}

/**
 * 4. 使用 Plotly.js 繪製圖表
 * (這是完全重寫的部分，取代您的 renderChart)
 * @param {object} data - 從 API 傳來的 data 物件
 * @param {object} description - 從 API 傳來的 description 物件
 */
function drawPlot(data, description) {
    const chartDiv = document.getElementById('plotly-chart');

    // --- 準備資料 (Traces) ---

    // 圖層 1：決策邊界背景 (使用 Heatmap)
    // 這會產生您 Colab 中的「紅色/藍色」背景区域
    const boundaryTrace = {
        type: 'heatmap',
        x: data.decision_boundary.xx[0], // X 軸網格 (1D)
        y: data.decision_boundary.yy.map(row => row[0]), // Y 軸網格 (1D)
        z: data.decision_boundary.Z, // 預測結果 (2D)
        colorscale: [
            ['0.0', 'rgb(74, 110, 184)'], // 0 (無心臟病) 的顏色 (藍色)
            ['1.0', 'rgb(187, 85, 85)']  // 1 (有心臟病) 的顏色 (紅色)
        ],
        zsmooth: false,
        showscale: false, // 隱藏旁邊的顏色條
        opacity: 0.6, // 設定透明度以匹配您 Colab 的 alpha=0.6
        name: '決策邊界'
    };

    // 建立一個小幫手函式，來過濾資料點
    const filterData = (points, y_val) => ({
        x: points.x1.filter((_, i) => points.y[i] === y_val),
        y: points.x2.filter((_, i) => points.y[i] === y_val)
    });

    // 過濾出 4 組資料
    const train_0 = filterData(data.train_points, 0); // 訓練 - 無病
    const train_1 = filterData(data.train_points, 1); // 訓練 - 有病
    const test_0 = filterData(data.test_points, 0);   // 測試 - 無病
    const test_1 = filterData(data.test_points, 1);   // 測試 - 有病

    // 圖層 2：訓練 - 無心臟病 (藍色 O)
    const traceTrain0 = {
        type: 'scatter', mode: 'markers',
        x: train_0.x, y: train_0.y,
        name: '無心臟病 (訓練)',
        marker: { color: 'blue', symbol: 'circle', size: 8, line: { color: 'black', width: 1 } }
    };

    // 圖層 3：訓練 - 有心臟病 (紅色 O)
    const traceTrain1 = {
        type: 'scatter', mode: 'markers',
        x: train_1.x, y: train_1.y,
        name: '有心臟病 (訓練)',
        marker: { color: 'red', symbol: 'circle', size: 8, line: { color: 'black', width: 1 } }
    };

    // 圖層 4：測試 - 無心臟病 (藍色 X)
    // (Plotly 使用 'cross' 來代表 'X')
    const traceTest0 = {
        type: 'scatter', mode: 'markers',
        x: test_0.x, y: test_0.y,
        name: '無心臟病 (測試)',
        marker: { color: 'blue', symbol: 'cross', size: 10, line: { color: 'black', width: 2 } }
    };

    // 圖層 5：測試 - 有心臟病 (紅色 X)
    const traceTest1 = {
        type: 'scatter', mode: 'markers',
        x: test_1.x, y: test_1.y,
        name: '有心臟病 (測試)',
        marker: { color: 'red', symbol: 'cross', size: 10, line: { color: 'black', width: 2 } }
    };

    // --- 組合所有圖層 ---
    const plotData = [
        boundaryTrace, // 背景
        traceTrain0,   // 散點
        traceTrain1,
        traceTest0,
        traceTest1
    ];

    // --- 版面配置 (Layout) ---
    const layout = {
        title: `Logistic Regression 決策邊界<br>(使用 ${description.x1_feature} 和 ${description.x2_feature})`,
        xaxis: {
            title: description.x1_feature, // X 軸標題
            zeroline: false
        },
        yaxis: {
            title: description.x2_feature, // Y 軸標題
            zeroline: false
        },
        hovermode: 'closest', // 滑鼠懸停效果
        legend: {
            traceorder: 'normal',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E2E2',
            borderwidth: 1
        },
        margin: { t: 80, b: 60, l: 60, r: 30 } // 邊距
    };

    // --- 繪製圖表 ---
    // 最後一步：呼叫 Plotly.newPlot
    Plotly.newPlot(chartDiv, plotData, layout, {
        responsive: true // 讓圖表自適應寬度
    });
    chartDiv.classList.remove('loading');
}