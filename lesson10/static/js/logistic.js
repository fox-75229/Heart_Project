/**
 * static/js/logistic.js
 * * (最終優化版 v3)
 * * 1. Jitter: 抖動資料
 * * 2. Tooltip: 顯示 是/否
 * * 3. Hover: 背景不觸發
 * * 4. 【修正】自訂 Modebar，只保留「下載」和「還原縮放」
 */

// 1. 等待 DOM 載入
document.addEventListener("DOMContentLoaded", () => {
    loadLogisticData();
});

// 2. 呼叫 API (此函式不變)
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

// 3. 更新評估指標 (不變)
function updateMetrics(metrics) {
    document.getElementById('metrics-accuracy').textContent = metrics.accuracy;
    document.getElementById('metrics-precision').textContent = metrics.precision;
    document.getElementById('metrics-recall').textContent = metrics.recall;
    document.getElementById('metrics-f1').textContent = metrics.f1;
    document.getElementById('metrics-auc').textContent = metrics.auc;
}

// 4. 使用 Plotly.js 繪製圖表
function drawPlot(data, description) {
    const chartDiv = document.getElementById('plotly-chart');
    const JITTER_AMOUNT = 0.15;

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

    // 準備資料 Traces (不變)
    const boundaryTrace = {
        type: 'heatmap', x: data.decision_boundary.xx[0],
        y: data.decision_boundary.yy.map(row => row[0]),
        z: data.decision_boundary.Z, colorscale: [['0.0', 'rgb(74, 110, 184)'], ['1.0', 'rgb(187, 85, 85)']],
        zsmooth: false, showscale: false, opacity: 0.6,
        name: '決策邊界', hoverinfo: 'skip'
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
    const plotData = [boundaryTrace, traceTrain0, traceTrain1, traceTest0, traceTest1];

    // 版面配置 (Layout) (不變)
    const layout = {
        title: `Logistic Regression 決策邊界<br>(使用 ${description.x1_feature} 和 ${description.x2_feature})`,
        xaxis: { title: description.x1_feature, zeroline: false },
        yaxis: { title: description.x2_feature, zeroline: false },
        hovermode: 'closest',
        legend: {
            traceorder: 'normal', bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E2E2', borderwidth: 1
        },
        margin: { t: 80, b: 60, l: 60, r: 30 }
    };

    // --- 【關鍵修正】 ---
    // 建立一個 config 物件來取代 displayModeBar: false
    const config = {
        responsive: true,

        // 移除您不想要的所有按鈕，只保留...
        // 'autoScale2d' (還原縮放 - 小屋按鈕)
        modeBarButtons: [
            ['autoScale2d']
            // 我們移除了: zoom2d, pan2d, zoomIn2d, zoomOut2d 等...
        ]
    };

    // 繪製圖表
    Plotly.newPlot(chartDiv, plotData, layout, config); // <-- 最後一個參數改為 config

    chartDiv.classList.remove('loading');
}