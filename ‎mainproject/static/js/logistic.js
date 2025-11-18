/**
 * 邏輯迴歸頁面腳本 (logistic.js)
 *
 * 負責 logistic.html 頁面的互動功能，特別是：
 * 1. 使用 Plotly.js 繪製 2D 決策邊界圖表。
 * 2. 處理使用者輸入並呼叫後端預測 API。
 * 3. 在圖表上以動畫形式顯示使用者的預測點 (星星)。
 */

// --- 全域變數 (用於儲存 Plotly 圖表狀態) ---
let globalPlotData = []; // 儲存圖表的所有圖層 (Traces)
let globalLayout = {};   // 儲存圖表佈局設定
let globalConfig = {};   // 儲存圖表互動設定
let GLOBAL_CHART_Y_MIN = 0; // 用於動畫起始位置計算

// --- 1. 頁面初始化 ---

document.addEventListener("DOMContentLoaded", () => {
    loadLogisticData();    // 載入資料並繪圖
    setupEventListeners(); // 綁定互動事件
});

// --- 2. 資料載入與繪圖 ---

/**
 * [非同步] 載入邏輯迴歸資料 (邊界數據、訓練/測試點、指標)
 * 呼叫 API: /api/logistic/data
 */
async function loadLogisticData() {
    const chartDiv = document.getElementById('plotly-chart');
    // 取得載入動畫元素 (稍後移除)
    const loadingSpinner = chartDiv.querySelector('.loading-spinner');
    const loadingText = chartDiv.querySelector('.loading-text');

    try {
        const response = await fetch('/api/logistic/data');
        if (!response.ok) throw new Error(`HTTP 錯誤! 狀態: ${response.status}`);

        const result = await response.json();

        if (result.success) {
            // 更新頁面上的資訊
            updateMetrics(result.metrics);
            updateModelInfo(result.description);
            // 繪製 Plotly 圖表
            drawPlot(result.data, result.description);

            // 移除載入動畫
            if (loadingSpinner) loadingSpinner.remove();
            if (loadingText) loadingText.remove();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error("載入邏輯迴歸資料失敗:", error);
        chartDiv.innerHTML = `<p class="error-text">❌ 載入圖表失敗: ${error.message}</p>`;
    }
}

/**
 * 綁定頁面互動事件
 */
function setupEventListeners() {
    const slider = document.getElementById('st-slider');
    const radios = document.querySelectorAll('input[name="angina_group"]');
    const predictBtn = document.getElementById('predict-btn');

    // 當輸入改變時，清除舊的預測結果 (文字與星星)
    if (slider) {
        slider.addEventListener('input', clearPrediction);
    }
    radios.forEach(radio => {
        radio.addEventListener('change', clearPrediction);
    });

    // 點擊預測按鈕
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }
}


// --- 3. Plotly.js 圖表繪製邏輯 ---

/**
 * 繪製主圖表
 * 包含：決策邊界 (Heatmap) + 訓練/測試資料點 (Scatter)
 */
function drawPlot(data, description) {
    const chartDiv = document.getElementById('plotly-chart');
    const JITTER_AMOUNT = 0.15; // 點的抖動幅度 (避免重疊)

    GLOBAL_CHART_Y_MIN = description.y_min || 0;

    // --- 輔助函式: 篩選數據並加入抖動 (Jitter) ---
    // 因為原始數據是離散的 (0, 1)，直接畫會全部疊在一起，
    // 加入隨機抖動可以讓點散開，便於觀察分佈密度。
    const filterAndJitterData = (points, y_val) => {
        const jittered_x = [];
        const jittered_y = [];
        const custom_data = [];

        for (let i = 0; i < points.y.length; i++) {
            if (points.y[i] === y_val) {
                const original_x = points.x1[i];
                const original_y = points.x2[i];
                const angina_string = (original_y === 1) ? '是' : '否';

                // 加入隨機抖動
                const x_jitter = (Math.random() - 0.5) * JITTER_AMOUNT * 2;
                const y_jitter = (Math.random() - 0.5) * JITTER_AMOUNT * 2;

                jittered_x.push(original_x + x_jitter);
                jittered_y.push(original_y + y_jitter);
                // 儲存原始數據供 Tooltip 使用
                custom_data.push({ x: original_x, y_str: angina_string });
            }
        }
        return { x: jittered_x, y: jittered_y, customdata: custom_data };
    };

    // --- 準備圖層 (Traces) ---

    // 1. 決策邊界 (背景熱圖)
    const boundaryTrace = {
        type: 'heatmap',
        x: data.decision_boundary.xx[0],
        y: data.decision_boundary.yy.map(row => row[0]),
        z: data.decision_boundary.Z,
        colorscale: [['0.0', 'rgb(74, 110, 184)'], ['1.0', 'rgb(187, 85, 85)']], // 藍到紅
        zsmooth: false,
        showscale: false,
        hoverinfo: 'skip',
        opacity: 0.6 // 半透明，讓點更清楚
    };

    // 2. 處理四組資料點 (訓練/測試 x 有病/沒病)
    const train_0 = filterAndJitterData(data.train_points, 0);
    const train_1 = filterAndJitterData(data.train_points, 1);
    const test_0 = filterAndJitterData(data.test_points, 0);
    const test_1 = filterAndJitterData(data.test_points, 1);

    // Tooltip 顯示模板
    const hover_template =
        '<b>%{data.name}</b><br><br>' +
        `<b>${description.x1_feature}:</b> %{customdata.x}<br>` +
        `<b>${description.x2_feature}:</b> %{customdata.y_str}<br>` +
        '<extra></extra>';

    // 定義資料點樣式
    const traceTrain0 = {
        type: 'scatter', mode: 'markers', x: train_0.x, y: train_0.y,
        customdata: train_0.customdata, hovertemplate: hover_template,
        name: '無心臟病 (訓練)', marker: { color: 'blue', symbol: 'circle', size: 8, line: { color: 'black', width: 1 } }
    };
    const traceTrain1 = {
        type: 'scatter', mode: 'markers', x: train_1.x, y: train_1.y,
        customdata: train_1.customdata, hovertemplate: hover_template,
        name: '有心臟病 (訓練)', marker: { color: 'red', symbol: 'circle', size: 8, line: { color: 'black', width: 1 } }
    };
    const traceTest0 = {
        type: 'scatter', mode: 'markers', x: test_0.x, y: test_0.y,
        customdata: test_0.customdata, hovertemplate: hover_template,
        name: '無心臟病 (測試)', marker: { color: 'blue', symbol: 'triangle-up', size: 10, line: { color: 'black', width: 1 } }
    };
    const traceTest1 = {
        type: 'scatter', mode: 'markers', x: test_1.x, y: test_1.y,
        customdata: test_1.customdata, hovertemplate: hover_template,
        name: '有心臟病 (測試)', marker: { color: 'red', symbol: 'triangle-up', size: 10, line: { color: 'black', width: 1 } }
    };

    // 將「基礎」資料存到全域變數 (不包含預測星星)
    globalPlotData = [boundaryTrace, traceTrain0, traceTrain1, traceTest0, traceTest1];

    // --- 設定圖表佈局 ---
    globalLayout = {
        title: ` ${description.x1_feature} 、 ${description.x2_feature}`,
        // 鎖定 X 軸範圍
        xaxis: {
            title: description.x1_feature,
            zeroline: false,
            range: [-0.75, 2.75],
            autorange: false
        },
        // 鎖定 Y 軸範圍
        yaxis: {
            title: description.x2_feature,
            zeroline: false,
            range: [-0.75, 1.75],
            autorange: false
        },
        hovermode: 'closest',
        // 圖例設定
        legend: {
            orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'center', x: 0.5,
        },
        margin: { t: 80, b: 60, l: 60, r: 30 }
    };

    globalConfig = {
        responsive: true,
        scrollZoom: false,
        modeBarButtons: [['autoScale2d']] // 只保留自動縮放按鈕
    };

    // 繪製圖表
    Plotly.newPlot(chartDiv, globalPlotData, globalLayout, globalConfig);
    chartDiv.classList.remove('loading');
}

// --- 4. 預測邏輯 ---

/**
 * [非同步] 處理預測請求
 * 1. 呼叫 API 獲取結果
 * 2. 更新文字
 * 3. 觸發星星動畫
 */
async function handlePrediction() {
    const predictBtn = document.getElementById('predict-btn');
    const loadingSpinner = document.getElementById('predict-loading');

    clearPrediction(); // 清除舊結果

    predictBtn.disabled = true;
    loadingSpinner.style.display = 'block';

    try {
        // 取得輸入並轉型
        const stSlope_str = document.getElementById('st-slider').value;
        const angina_str = document.querySelector('input[name="angina_group"]:checked').value;
        const stSlope = parseFloat(stSlope_str);
        const angina = parseFloat(angina_str);

        // 呼叫 API
        const response = await fetch('/api/logistic/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ st_slope: stSlope, angina: angina })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "預測失敗");
        }

        const result = await response.json();

        if (result.success) {
            // 更新文字建議
            updatePredictionText(stSlope_str, angina_str);

            // 顯示星星動畫 (傳入座標與預測類別)
            showPredictionMarker(stSlope, angina, result.prediction_class);
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error("預測時發生錯誤:", error);
        const container = document.getElementById('predict-container');
        const textElement = container.querySelector('.predict-text p');
        if (textElement) {
            textElement.innerHTML = `<span class="error-text">❌ 預測失敗: ${error.message}</span>`;
        }
        container.style.display = 'block';
    } finally {
        predictBtn.disabled = false;
        loadingSpinner.style.display = 'none';
    }
}

/**
 * 根據輸入值更新預測文字與建議
 * (註: 這裡的邏輯是寫死的，建議未來可改用後端回傳的資訊)
 */
function updatePredictionText(stSlope, angina) {
    const container = document.getElementById('predict-container');
    const textElement = container.querySelector('.predict-text p');

    let riskLevel = "";
    let suggestionText = "";
    let riskClass = "";

    // 簡單的規則判斷
    if (stSlope == 0 && angina == 0) {
        riskLevel = "中"; riskClass = "risk-medium"; suggestionText = "可能有無症狀心肌缺血的風險 建議就診";
    } else if ((stSlope == 0 && angina == 1) || (stSlope == 1 && angina == 1)) {
        riskLevel = "高"; riskClass = "risk-high"; suggestionText = "患有心臟病風險極高 建議即刻就診進行檢查";
    } else if (stSlope == 1 && angina == 0) {
        riskLevel = "中"; riskClass = "risk-medium"; suggestionText = "可能有無症狀心肌缺血的風險 若患有糖尿病或高齡個體 建議就診";
    } else if (stSlope == 2 && angina == 0) {
        riskLevel = "低"; riskClass = "risk-low"; suggestionText = "健康個體，身體狀況良好";
    } else if (stSlope == 2 && angina == 1) {
        riskLevel = "中"; riskClass = "risk-medium"; suggestionText = "身體狀況良好 痛感或許來自肌肉拉傷";
    } else {
        riskLevel = "未知"; riskClass = "risk-medium"; suggestionText = "無法判斷，請確認輸入。";
    }

    textElement.innerHTML = `
        經過預測:本案例屬於:
        <strong id="predict-risk-level" class="deg ${riskClass}">${riskLevel}</strong>
        風險族群
        <strong id="predict-suggestion" class="sug">${suggestionText}</strong>
    `;
    container.style.display = 'flex';
}

// --- 5. 動畫與視覺效果 ---

/**
 * 在圖表上顯示預測結果 (星星)，並帶有入場動畫
 */
function showPredictionMarker(x, y, predictionClass) {
    const chartDiv = document.getElementById('plotly-chart');
    const markerColor = (predictionClass === 1) ? '#ff4800ff' : '#0392ffff'; // 紅色或藍色

    // 2. 設定起始位置 (從畫面下方飛入)
    const startY = GLOBAL_CHART_Y_MIN - 0.8;

    // 定義星星圖層
    const tracePrediction = {
        type: 'scatter',
        mode: 'markers',
        name: '您的預測',
        x: [x],
        y: [startY], // 初始位置
        hoverinfo: 'skip',
        marker: {
            symbol: 'star',
            size: 25,
            color: markerColor,
            line: { color: 'black', width: 2 }
        }
    };

    // 3. 將星星加入圖表數據 (作為新的 Trace)
    const newPlotData = [...globalPlotData, tracePrediction];

    // 重新繪製圖表 (Plotly.newPlot 比 Plotly.react 更穩定)
    Plotly.newPlot(chartDiv, newPlotData, globalLayout, globalConfig);

    // 4. 執行上升動畫
    let currentY = startY;
    const duration = 500; // 動畫總時長 (毫秒)
    const fps = 60;
    const step = (y - startY) / (duration / (1000 / fps)); // 每一禎移動的距離

    const interval = setInterval(() => {
        currentY += step;

        // 檢查是否到達目標位置
        if ((step > 0 && currentY >= y) || (step < 0 && currentY <= y)) {
            currentY = y;
            clearInterval(interval); // 停止動畫
        }

        // 使用 Plotly.restyle 僅更新星星的位置 (這是最後一個 Trace)
        Plotly.restyle(chartDiv, { y: [[currentY]] }, newPlotData.length - 1);
    }, 1000 / fps);
}

/**
 * 清除預測結果 (重置圖表)
 */
function clearPrediction() {
    // 隱藏文字區塊
    const container = document.getElementById('predict-container');
    if (container) container.style.display = 'none';

    const chartDiv = document.getElementById('plotly-chart');

    // 如果圖表數據存在，重繪圖表 (會自動移除星星 Trace，因為 globalPlotData 裡沒有星星)
    if (globalPlotData && globalPlotData.length > 0) {
        Plotly.newPlot(chartDiv, globalPlotData, globalLayout, globalConfig);
    }
}

// --- 6. UI 更新輔助函式 ---

function updateMetrics(metrics) {
    document.getElementById('metrics-accuracy').textContent = metrics.accuracy;
    document.getElementById('metrics-precision').textContent = metrics.precision;
    document.getElementById('metrics-recall').textContent = metrics.recall;
    document.getElementById('metrics-f1').textContent = metrics.f1;
    document.getElementById('metrics-auc').textContent = metrics.auc;
}

function updateModelInfo(description) {
    document.getElementById('desc-dataset').textContent = description.dataset;
    document.getElementById('desc-total-samples').textContent = description.total_samples;
    document.getElementById('desc-train-size').textContent = description.train_size;
    document.getElementById('desc-test-size').textContent = description.test_size;
    document.getElementById('desc-target').textContent = description.target;
    document.getElementById('desc-features').textContent = description.selected_features.join(', ');
}