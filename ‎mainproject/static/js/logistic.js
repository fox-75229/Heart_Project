/**
 * static/js/logistic.js
 * * (最終優化版 v11 - 修正圖例到「頂部」)
 * * 1. 【修正】: 將圖例 (Legend) y: 設為 1.02, yanchor: 設為 'bottom'
 * * -> 這會將圖例放在「標題」和「圖表」之間
 * * 2. 保持 Heatmap, 十字游標, 禁用滾輪, 只保留還原按鈕
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
            updateModelInfo(result.description);

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

// 更新模型評估指標 (Accuracy, Precision, Recall, F1, AUC)
function updateMetrics(metrics) {
    document.getElementById('metrics-accuracy').textContent = metrics.accuracy;
    document.getElementById('metrics-precision').textContent = metrics.precision;
    document.getElementById('metrics-recall').textContent = metrics.recall;
    document.getElementById('metrics-f1').textContent = metrics.f1;
    document.getElementById('metrics-auc').textContent = metrics.auc;
}

// 更新模型基本資訊 (資料集、樣本數、特徵等)
function updateModelInfo(description) {
    document.getElementById('desc-dataset').textContent = description.dataset;
    document.getElementById('desc-total-samples').textContent = description.total_samples;
    document.getElementById('desc-train-size').textContent = description.train_size;
    document.getElementById('desc-test-size').textContent = description.test_size;
    document.getElementById('desc-target').textContent = description.target;
    // 將特徵陣列轉為字串顯示
    document.getElementById('desc-features').textContent = description.selected_features.join(', ');
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

    // --- 準備圖層 (Traces) --- (不變)
    const boundaryTrace = {
        type: 'heatmap',
        x: data.decision_boundary.xx[0], // 1D X 軸
        y: data.decision_boundary.yy.map(row => row[0]), // 1D Y 軸
        z: data.decision_boundary.Z, // 2D Z 值
        colorscale: [
            ['0.0', 'rgb(74, 110, 184)'], // 藍色
            ['1.0', 'rgb(187, 85, 85)']  // 紅色
        ],
        zsmooth: false, // <-- 確保是階梯狀
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
    const plotData = [boundaryTrace, traceTrain0, traceTrain1, traceTest0, traceTest1];

    // --- 【關鍵修正：版面配置】 ---
    const layout = {
        title: ` ${description.x1_feature} 、 ${description.x2_feature}`,
        xaxis: { title: description.x1_feature, zeroline: false },
        yaxis: { title: description.x2_feature, zeroline: false },
        hovermode: 'closest',

        //
        legend: {
            orientation: 'h',     // 'h' = 水平排列
            yanchor: 'bottom',    // 錨點在圖例的「底部」
            y: 1.02,              // <-- 【修正】 1.02 = 放在繪圖區的「正上方」
            xanchor: 'center',    // 錨點在圖例的「中心」
            x: 0.5,               // 位置在 X 軸 50% (中心)

            // itemdoubleclick: false, // 禁用雙擊圖例項目
        },
        // 
        margin: {
            t: 80, // top (保持 80, 為標題和新圖例騰出空間)
            b: 60, // bottom (改回 60 即可)
            l: 60, // left
            r: 30  // right
        }
    };

    // 圖表互動設定 (不變)
    const config = {
        responsive: true,
        scrollZoom: false,  // 禁用滾輪縮放
        modeBarButtons: [
            ['autoScale2d'] // 只保留「還原縮放」(小屋按鈕)
        ]
    };

    // 繪製圖表
    Plotly.newPlot(chartDiv, plotData, layout, config);
    chartDiv.classList.remove('loading');

    
}