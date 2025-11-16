// static/js/logistic.js (v2 - 互動預測版)

// 【新增】全域變數，用於儲存圖表的 Y 軸最小值，供動畫使用
let GLOBAL_CHART_Y_MIN = 0;

// 1. 等待 DOM 載入
document.addEventListener("DOMContentLoaded", () => {
    loadLogisticData();
    setupEventListeners(); // 【新增】 綁定控制項
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

            // 【修改】drawPlot 現在會儲存 y_min
            drawPlot(result.data, result.description);

            if (loadingSpinner) loadingSpinner.remove();
            if (loadingText) loadingText.remove();
        }
        else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error("載入邏輯迴歸資料失敗:", error);
        chartDiv.innerHTML = `<p class="error-text">❌ 載入圖表失敗: ${error.message}</p>`;
    }
}

// 3. 綁定所有事件監聽 (同 decision_tree.js)
function setupEventListeners() {
    // A. ST 段斜率滑桿
    const slider = document.getElementById('st-slider');
    const kValue = document.getElementById('st-value');
    if (slider) {
        slider.addEventListener('input', () => {
            kValue.textContent = slider.value;
            clearPredictionStar(); // 【修改】只清除星星
            document.getElementById('predict-container').style.display = 'none'; // 【新增】隱藏文字
        });
    }

    // B. 運動心絞痛 Radio
    const radios = document.querySelectorAll('input[name="angina_group"]');
    radios.forEach(radio => {
        radio.addEventListener('change', () => {
            clearPredictionStar(); // 【修改】只清除星星
            document.getElementById('predict-container').style.display = 'none'; // 【新增】隱藏文字
        });
    });

    // C. 預測按鈕
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }
}

// 4. 使用 Plotly.js 繪製圖表
function drawPlot(data, description) {
    const chartDiv = document.getElementById('plotly-chart');
    const JITTER_AMOUNT = 0.15;

    // 【【【新增】】】
    // 儲存 Y 軸最小值到全域變數，供動畫使用
    GLOBAL_CHART_Y_MIN = description.y_min || 0;

    // Jitter 函式 (不變)
    const filterAndJitterData = (points, y_val) => {
        // ... (你原有的 Jitter 函式 - 不變) ...
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

    // --- 準備圖層 (Traces) --- (不變)
    const boundaryTrace = {
        type: 'heatmap',
        x: data.decision_boundary.xx[0],
        y: data.decision_boundary.yy.map(row => row[0]),
        z: data.decision_boundary.Z,
        colorscale: [
            ['0.0', 'rgb(74, 110, 184)'], // 藍色
            ['1.0', 'rgb(187, 85, 85)']  // 紅色
        ],
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

    // 【修改】 將 plotData 設為全域變數，以便我們新增/刪除 trace
    // (移除 const)
    plotData = [boundaryTrace, traceTrain0, traceTrain1, traceTest0, traceTest1];

    // --- 版面配置 --- (不變)
    const layout = {
        title: ` ${description.x1_feature} 、 ${description.x2_feature}`,
        xaxis: { title: description.x1_feature, zeroline: false },
        yaxis: { title: description.x2_feature, zeroline: false },
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

    // 圖表互動設定 (不變)
    const config = {
        responsive: true,
        scrollZoom: false,
        modeBarButtons: [
            ['autoScale2d']
        ]
    };

    // 繪製圖表
    Plotly.newPlot(chartDiv, plotData, layout, config);
    chartDiv.classList.remove('loading');
}


// --- 【【【以下為新增的預測功能】】】 ---

// 5. 【新增】處理預測
async function handlePrediction() {
    const predictBtn = document.getElementById('predict-btn');
    const loadingSpinner = document.getElementById('predict-loading');

    // 【【【關鍵修改】】】
    // 1. 先清除舊的「星星」(如果有的話)
    clearPredictionStar();
    // 2. 隱藏舊的「文字」
    document.getElementById('predict-container').style.display = 'none';

    // 顯示載入中
    predictBtn.disabled = true;
    loadingSpinner.style.display = 'block';

    try {
        // A. 獲取輸入值
        const stSlope = document.getElementById('st-slider').value;
        const angina = document.querySelector('input[name="angina_group"]:checked').value;

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

            // C. 更新預測文字 (這會讓 container.style.display = 'block')
            updatePredictionText(stSlope, angina);

            // D. 【星星動畫】
            animatePredictionMarker(stSlope, angina, result.prediction_class);

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

// 6. 【新增】更新預測文字 (同 decision_tree.js)
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

// 7. 【新增】星星動畫
function animatePredictionMarker(x, y, predictionClass) {
    const chartDiv = document.getElementById('plotly-chart');

    // 【【【關鍵修改】】】
    // (清除邏輯已移到 handlePrediction，這裡不再呼叫)
    // clearPrediction(); // <-- 移除這行

    // 根據預測結果決定星星顏色 (0=藍, 1=紅)
    const markerColor = (predictionClass === 1) ? 'red' : 'blue';

    // 1. 定義星星的「起始」狀態 (在圖表底部)
    const startTrace = {
        x: [x],
        y: [GLOBAL_CHART_Y_MIN], // 從 Y 軸底部開始
        mode: 'markers',
        type: 'scatter',
        name: '您的預測',
        hoverinfo: 'skip',
        marker: {
            symbol: 'star',
            size: 25,
            color: markerColor,
            line: {
                color: 'black',
                width: 2
            }
        }
    };

    // 2. 將星星 (在底部) 新增到圖表
    Plotly.addTraces(chartDiv, startTrace);

    // 3. 取得這個新 Trace 的索引 (它是最後一個)
    const traceIndex = chartDiv.data.length - 1;

    // 4. 定義動畫的「結束」狀態
    const endFrame = {
        data: [{
            y: [y] // Y 軸移動到「真正」的位置 (0 或 1)
        }]
    };

    // 5. 執行動畫
    Plotly.animate(chartDiv, endFrame, {
        frame: {
            duration: 700, // 動畫時間 (毫秒)
            redraw: false
        },
        transition: {
            duration: 700,
            easing: 'cubic-out' // 緩動效果
        }
    });
}

// 8. 【新增】清除預測 (清除星星和文字)
function clearPredictionStar() {
    // B. 移除星星 (Trace)
    const chartDiv = document.getElementById('plotly-chart');
    if (chartDiv.data) {
        // 檢查最後一個 trace 是否為 '您的預測'
        const lastTrace = chartDiv.data[chartDiv.data.length - 1];
        if (lastTrace && lastTrace.name === '您的預測') {
            // 只刪除最後一個 trace
            Plotly.deleteTraces(chartDiv, -1);
        }
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