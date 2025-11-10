/**
 * static/js/decision_tree.js
 * * (這是 logistic.js v8 的複製品, 只是 API 路徑不同)
 * 1. 呼叫 API (/api/decision_tree/data)
 * 2. 更新指標卡
 * 3. 繪製 Plotly 決策邊界圖 (Heatmap + Scatter)
 */

// 1. 頁面載入完成後，執行 loadData
document.addEventListener("DOMContentLoaded", () => {
    loadData();
});

// 2. 呼叫 API，取得圖表和指標資料
async function loadData() { // (重新命名函式)
    const chartDiv = document.getElementById('plotly-chart');
    const loadingSpinner = chartDiv.querySelector('.loading-spinner');
    const loadingText = chartDiv.querySelector('.loading-text');

    try {
        // --- 【關鍵修改】 ---
        // 呼叫「決策樹」的 API
        const response = await fetch('/api/decision_tree/data');
        // ------------------

        if (!response.ok) {
            throw new Error(`HTTP 錯誤! 狀態: ${response.status}`);
        }
        const result = await response.json();

        // 成功取得資料
        if (result.success) {
            updateMetrics(result.metrics); // 更新指標
            drawPlot(result.data, result.description); // 繪製圖表

            // 移除讀取中提示
            if (loadingSpinner) loadingSpinner.remove();
            if (loadingText) loadingText.remove();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        // 處理所有錯誤
        console.error("載入決策樹資料失敗:", error);
        chartDiv.innerHTML = `<p class="error-text">❌ 載入圖表失敗: ${error.message}</p>`;
    }
}

// 3. 將 API 回傳的指標填入 HTML (不變)
function updateMetrics(metrics) {
    document.getElementById('metrics-accuracy').textContent = metrics.accuracy;
    document.getElementById('metrics-precision').textContent = metrics.precision;
    document.getElementById('metrics-recall').textContent = metrics.recall;
    document.getElementById('metrics-f1').textContent = metrics.f1;
    document.getElementById('metrics-auc').textContent = metrics.auc;
}

// 4. 使用 Plotly.js 繪製圖表 (不變)
function drawPlot(data, description) {
    const chartDiv = document.getElementById('plotly-chart');
    const JITTER_AMOUNT = 0.15; // 抖動幅度

    // 幫手函式: 篩選資料點 (y=0/1)，並加入抖動
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

    // 圖層 1: 背景 (決策邊界)
    const boundaryTrace = {
        type: 'heatmap',
        x: data.decision_boundary.xx[0],
        y: data.decision_boundary.yy.map(row => row[0]),
        z: data.decision_boundary.Z,
        colorscale: [
            ['0.0', 'rgb(74, 110, 184)'], // 藍色 (0)
            ['1.0', 'rgb(187, 85, 85)']  // 紅色 (1)
        ],
        zsmooth: false, // 階梯狀
        showscale: false,
        hoverinfo: 'skip',
        opacity: 0.6
    };

    // 準備 4 組抖動後的資料點
    const train_0 = filterAndJitterData(data.train_points, 0);
    const train_1 = filterAndJitterData(data.train_points, 1);
    const test_0 = filterAndJitterData(data.test_points, 0);
    const test_1 = filterAndJitterData(data.test_points, 1);

    // 自訂 tooltip 顯示的內容
    const hover_template =
        '<b>%{data.name}</b><br><br>' +
        `<b>${description.x1_feature}:</b> %{customdata.x}<br>` +
        `<b>${description.x2_feature}:</b> %{customdata.y_str}<br>` +
        '<extra></extra>';

    // 散點圖 (不變)
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

    // 圖表版面配置
    const layout = {
        // 【修改】標題
        title: `Decision Tree 決策邊界<br>(使用 ${description.x1_feature} 和 ${description.x2_feature})`,
        xaxis: { title: description.x1_feature, zeroline: false },
        yaxis: { title: description.x2_feature, zeroline: false },
        hovermode: 'closest',
        // (我們還原到使用預設的 top-right 圖例)
        legend: {
            traceorder: 'normal', bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E2E2', borderwidth: 1
        },
        margin: { t: 80, b: 60, l: 60, r: 30 }
    };

    // 圖表互動設定 (使用您 logistic 頁面的最終穩定版)
    const config = {
        responsive: true,
        scrollZoom: false,  // 禁用滾輪縮放
        modeBarButtons: [
            ['zoom2d', 'autoScale2d'] // 保留「框選」和「還原」
        ]
    };

    // 繪製圖表
    Plotly.newPlot(chartDiv, plotData, layout, config);

    // 移除 loading class
    chartDiv.classList.remove('loading');
}