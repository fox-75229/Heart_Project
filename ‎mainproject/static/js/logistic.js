// static/js/logistic.js 

// 建立全域變數來儲存圖表狀態
let globalPlotData = [];
let globalLayout = {};
let globalConfig = {};
let GLOBAL_CHART_Y_MIN = 0; // (這個我們還是保留)

// 1. 等待 DOM 載入
document.addEventListener("DOMContentLoaded", () => {
    loadLogisticData();
    setupEventListeners();
});

// 2. 呼叫 API (不變)
async function loadLogisticData() {
    const chartDiv = document.getElementById('plotly-chart');
    const loadingSpinner = chartDiv.querySelector('.loading-spinner');
    const loadingText = chartDiv.querySelector('.loading-text');

    try {
        const response = await fetch('/api/logistic/data');
        if (!response.ok) {
            throw new Error(`HTTP 錯誤! 狀態: ${response.status}`);
        }
        const result = await response.json();

        if (result.success) {
            updateMetrics(result.metrics);
            updateModelInfo(result.description);
            drawPlot(result.data, result.description);

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

// 3. 綁定所有事件監聽 (v3 - 修正版)
function setupEventListeners() {
    // A. ST 段斜率滑桿
    const slider = document.getElementById('st-slider');
    
    if (slider) {
        slider.addEventListener('input', () => {
            clearPrediction(); //呼叫 清除函式
        });
    }

    // B. 運動心絞痛 Radio
    const radios = document.querySelectorAll('input[name="angina_group"]');
    radios.forEach(radio => {
        radio.addEventListener('change', () => {
            clearPrediction(); //呼叫 清除函式
        });
    });

    // C. 預測按鈕
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }
}


// -----------------------------------------------------
// 4. 使用 Plotly.js 繪製圖表 (儲存狀態)
// -----------------------------------------------------
function drawPlot(data, description) {
    const chartDiv = document.getElementById('plotly-chart');
    const JITTER_AMOUNT = 0.15;

    GLOBAL_CHART_Y_MIN = description.y_min || 0;

    // Jitter 函式 (不變)
    const filterAndJitterData = (points, y_val) => {
        const jittered_x = [];
        const jittered_y = [];
        const custom_data = [];
        for (let i = 0; i < points.y.length; i++) {
            if (points.y[i] === y_val) {
                const original_x = points.x1[i];
                const original_y = points.x2[i];
                const angina_string = (original_y === 1) ? '是' : '否';
                const x_jitter = (Math.random() - 0.5) * JITTER_AMOUNT * 2;
                const y_jitter = (Math.random() - 0.5) * JITTER_AMOUNT * 2;
                jittered_x.push(original_x + x_jitter);
                jittered_y.push(original_y + y_jitter);
                custom_data.push({ x: original_x, y_str: angina_string });
            }
        }
        return { x: jittered_x, y: jittered_y, customdata: custom_data };
    };

    // --- 準備圖層 (Traces) ---
    const boundaryTrace = {
        type: 'heatmap',
        x: data.decision_boundary.xx[0],
        y: data.decision_boundary.yy.map(row => row[0]),
        z: data.decision_boundary.Z,
        colorscale: [['0.0', 'rgb(74, 110, 184)'], ['1.0', 'rgb(187, 85, 85)']],
        zsmooth: false,
        showscale: false,
        hoverinfo: 'skip',
        opacity: 0.6
    };

    const train_0 = filterAndJitterData(data.train_points, 0);
    const train_1 = filterAndJitterData(data.train_points, 1);
    const test_0 = filterAndJitterData(data.test_points, 0);
    const test_1 = filterAndJitterData(data.test_points, 1);

    const hover_template =
        '<b>%{data.name}</b><br><br>' +
        `<b>${description.x1_feature}:</b> %{customdata.x}<br>` +
        `<b>${description.x2_feature}:</b> %{customdata.y_str}<br>` +
        '<extra></extra>';

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

    // 1. 將「原始」資料存到全域變數
    globalPlotData = [boundaryTrace, traceTrain0, traceTrain1, traceTest0, traceTest1];

    // --- 【【【v12.1 關鍵修正：鎖定 X/Y 軸】】】 ---
    globalLayout = {
        title: ` ${description.x1_feature} 、 ${description.x2_feature}`,

        // 【修改】 鎖定 X 軸
        xaxis: {
            title: description.x1_feature,
            zeroline: false,
            range: [-0.75, 2.75], // <-- 給予固定範圍
            autorange: false     // <-- 關閉自動縮放
        },

        // 【修改】 鎖定 Y 軸
        yaxis: {
            title: description.x2_feature,
            zeroline: false,
            range: [-0.75, 1.75], // <-- 給予固定範圍
            autorange: false     // <-- 關閉自動縮放
        },

        hovermode: 'closest',
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: 1.02,
            xanchor: 'center',
            x: 0.5,
        },
        margin: { t: 80, b: 60, l: 60, r: 30 }
    };

    // 3. 將「原始」設定存到全域變數
    globalConfig = {
        responsive: true,
        scrollZoom: false,
        modeBarButtons: [
            ['autoScale2d']
        ]
    };

    // 4. 繪製「原始」圖表
    Plotly.newPlot(chartDiv, globalPlotData, globalLayout, globalConfig);
    chartDiv.classList.remove('loading');
}

// -----------------------------------------------------
// 5. 【核心】處理預測 (呼叫 showPredictionMarker)
// -----------------------------------------------------
async function handlePrediction() {
    const predictBtn = document.getElementById('predict-btn');
    const loadingSpinner = document.getElementById('predict-loading');

    //呼叫新的清除函式
    clearPrediction();

    // 顯示載入中
    predictBtn.disabled = true;
    loadingSpinner.style.display = 'block';

    try {
        // A. 獲取輸入值 (字串)
        const stSlope_str = document.getElementById('st-slider').value;
        const angina_str = document.querySelector('input[name="angina_group"]:checked').value;

        // 【v3 修正】轉型
        const stSlope = parseFloat(stSlope_str);
        const angina = parseFloat(angina_str);

        // B. 呼叫新的預測 API
        const response = await fetch('/api/logistic/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                st_slope: stSlope,
                angina: angina
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "預測失敗");
        }

        const result = await response.json();

        if (result.success) {

            // C. 更新預測文字
            updatePredictionText(stSlope_str, angina_str);

            // D. 顯示預測標記
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
        // 隱藏載入中
        predictBtn.disabled = false;
        loadingSpinner.style.display = 'none';
    }
}

// -----------------------------------------------------
// 6. 更新預測文字 (v9 - 不變)
// -----------------------------------------------------
function updatePredictionText(stSlope, angina) {
    const container = document.getElementById('predict-container');
    const textElement = container.querySelector('.predict-text p');

    let riskLevel = "";
    let suggestionText = "";
    let riskClass = "";

    // --- 使用 6 種輸入組合 ---
    if (stSlope == 0 && angina == 0) {
        riskLevel = "中";
        riskClass = "risk-medium";
        suggestionText = "可能有無症狀心肌缺血的風險 建議就診";
    }
    else if (stSlope == 0 && angina == 1) {
        riskLevel = "高";
        riskClass = "risk-high";
        suggestionText = "患有心臟病風險極高 建議即刻就診進行檢查";
    }
    else if (stSlope == 1 && angina == 0) {
        riskLevel = "中";
        riskClass = "risk-medium";
        suggestionText = "可能有無症狀心肌缺血的風險 若患有糖尿病或高齡個體 建議就診";
    }
    else if (stSlope == 1 && angina == 1) {
        riskLevel = "高";
        riskClass = "risk-high";
        suggestionText = "患有心臟病風險極高 建議即刻就診進行檢查";
    }
    else if (stSlope == 2 && angina == 0) {
        riskLevel = "低";
        riskClass = "risk-low";
        suggestionText = "健康個體，身體狀況良好";
    }
    else if (stSlope == 2 && angina == 1) {
        riskLevel = "中";
        riskClass = "risk-medium";
        suggestionText = "身體狀況良好 痛感或許來自肌肉拉傷";
    }
    else {
        riskLevel = "未知";
        riskClass = "risk-medium";
        suggestionText = "無法判斷，請確認輸入。";
    }

    // 注入 HTML
    textElement.innerHTML = `
        經過預測:本案例屬於:
        <strong id="predict-risk-level" class="deg ${riskClass}">${riskLevel}</strong>
        風險族群
        <strong id="predict-suggestion" class="sug">${suggestionText}</strong>
    `;

    // 顯示結果
    container.style.display = 'block';
}

// -----------------------------------------------------
// 7. 顯示星星 (使用 newPlot)
// -----------------------------------------------------
// --- 輔助：取得目前 Y 軸顯示範圍（安全取值） ---
function getYAxisRange(chartDiv) {
    // 優先使用已渲染圖表的 _fullLayout
    try {
        if (chartDiv && chartDiv._fullLayout && chartDiv._fullLayout.yaxis && Array.isArray(chartDiv._fullLayout.yaxis.range)) {
            return chartDiv._fullLayout.yaxis.range;
        }
    } catch (e) {
        // 忽略
    }

    // 若 globalLayout 有預設 range 則使用
    if (globalLayout && globalLayout.yaxis && Array.isArray(globalLayout.yaxis.range)) {
        return globalLayout.yaxis.range;
    }

    // 最後 fallback：使用預設最小值與最小+1
    return [GLOBAL_CHART_Y_MIN, GLOBAL_CHART_Y_MIN + 1];
}

// --- 新的 showPredictionMarker（加入自下而上入場動畫） ---
function showPredictionMarker(x, y, predictionClass) {
    const chartDiv = document.getElementById('plotly-chart');
    const markerColor = (predictionClass === 1) ? '#ff4800ff' : '#0392ffff';

    // 1. 先將舊星星清除
    clearPrediction();

    // 2. 建立星星 trace（起始位置在下面)
    const startY = GLOBAL_CHART_Y_MIN - 0.8;

    const tracePrediction = {
        type: 'scatter',
        mode: 'markers',
        name: '您的預測',
        x: [x],
        y: [startY], // ← 從下面開始
        hoverinfo: 'skip',
        marker: {
            symbol: 'star',
            size: 25,
            color: markerColor,
            line: { color: 'black', width: 2 }
        }
    };

    // 3. 新圖層（加入星星）
    const newPlotData = [...globalPlotData, tracePrediction];

    // 4. 重新繪製（保留主圖 + 星星的初始位置）
    Plotly.newPlot(chartDiv, newPlotData, globalLayout, globalConfig);

    // 5. —— 動畫 ——（從 startY 動到 y）
    let currentY = startY;
    const duration = 500; // 動畫時間 (ms)
    const fps = 60;
    const step = (y - startY) / (duration / (1000 / fps));

    const interval = setInterval(() => {
        currentY += step;

        if ((step > 0 && currentY >= y) || (step < 0 && currentY <= y)) {
            currentY = y;  // 最終位置
            clearInterval(interval);
        }

        // 更新星星位置（星星是最後一個 trace）
        Plotly.restyle(chartDiv, { y: [[currentY]] }, newPlotData.length - 1);
    }, 1000 / fps);
}




// -----------------------------------------------------
// 8. 清除預測 (使用 newPlot)
// -----------------------------------------------------
function clearPrediction() {
    // 隱藏預測文字
    const container = document.getElementById('predict-container');
    if (container) container.style.display = 'none';

    const chartDiv = document.getElementById('plotly-chart');

    // 重新繪製原始 5 個 trace（去掉星星）
    if (globalPlotData && globalPlotData.length > 0) {
        Plotly.newPlot(chartDiv, globalPlotData, globalLayout, globalConfig);
    }
}



// --- (舊的) 填充指標 (不變) ---
function updateMetrics(metrics) {
    document.getElementById('metrics-accuracy').textContent = metrics.accuracy;
    document.getElementById('metrics-precision').textContent = metrics.precision;
    document.getElementById('metrics-recall').textContent = metrics.recall;
    document.getElementById('metrics-f1').textContent = metrics.f1;
    document.getElementById('metrics-auc').textContent = metrics.auc;
}

// --- (舊的) 填充模型資訊 (不變) ---
function updateModelInfo(description) {
    document.getElementById('desc-dataset').textContent = description.dataset;
    document.getElementById('desc-total-samples').textContent = description.total_samples;
    document.getElementById('desc-train-size').textContent = description.train_size;
    document.getElementById('desc-test-size').textContent = description.test_size;
    document.getElementById('desc-target').textContent = description.target;
    document.getElementById('desc-features').textContent = description.selected_features.join(', ');
}