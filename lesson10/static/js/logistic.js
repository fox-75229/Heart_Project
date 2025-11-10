

// 1. 等待 DOM 載入
document.addEventListener("DOMContentLoaded", () => {
    loadLogisticData();
});

// 2. 呼叫 API
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

    // 【優化】建立新的「抖動+儲存原始資料」函式
    const filterAndJitterData = (points, y_val) => {
        const jittered_x = [];
        const jittered_y = [];
        const custom_data = []; // 儲存原始資料

        for (let i = 0; i < points.y.length; i++) {
            if (points.y[i] === y_val) {
                // 原始資料
                const original_x = points.x1[i]; // ST 段斜率 (0, 1, 2)
                const original_y = points.x2[i]; // 運動是否誘發心絞痛 (0, 1)

                // --- 【關鍵修正 1：映射 0/1 為 '否'/'是'】 ---
                const angina_string = (original_y === 1) ? '是' : '否';

                // 抖動
                const x_jitter = (Math.random() - 0.5) * JITTER_AMOUNT * 2;
                const y_jitter = (Math.random() - 0.5) * JITTER_AMOUNT * 2;

                jittered_x.push(original_x + x_jitter);
                jittered_y.push(original_y + y_jitter);

                // 儲存原始資料，用於 tooltip
                custom_data.push({
                    x: original_x,
                    y: original_y,
                    y_str: angina_string // <-- 儲存映射後的字串
                });
            }
        }
        return { x: jittered_x, y: jittered_y, customdata: custom_data };
    };

    // --- 準備資料 (Traces) ---

    // 圖層 1：決策邊界背景 (不變)
    const boundaryTrace = {
        type: 'heatmap',
        x: data.decision_boundary.xx[0],
        y: data.decision_boundary.yy.map(row => row[0]),
        z: data.decision_boundary.Z,
        colorscale: [
            ['0.0', 'rgb(74, 110, 184)'],
            ['1.0', 'rgb(187, 85, 85)']
        ],
        zsmooth: false, showscale: false, opacity: 0.6,
        name: '決策邊界',
        hoverinfo: 'skip' // 不觸發懸停
    };

    // 呼叫新函式來取得抖動後的資料
    const train_0 = filterAndJitterData(data.train_points, 0);
    const train_1 = filterAndJitterData(data.train_points, 1);
    const test_0 = filterAndJitterData(data.test_points, 0);
    const test_1 = filterAndJitterData(data.test_points, 1);

    // --- 【關鍵修正 2：更新 Tooltip 模板】 ---
    const hover_template =
        '<b>%{data.name}</b><br><br>' + // 顯示圖例名稱 (e.g., 無心臟病 (訓練))
        `<b>${description.x1_feature}:</b> %{customdata.x}<br>` + // 顯示 ST 段斜率 (0, 1, 2)
        `<b>${description.x2_feature}:</b> %{customdata.y_str}<br>` + // <-- 顯示 '是' / '否'
        '<extra></extra>'; // 隱藏額外的 "trace" 訊息

    // 圖層 2：訓練 - 無心臟病 (藍色 O)
    const traceTrain0 = {
        type: 'scatter', mode: 'markers',
        x: train_0.x, y: train_0.y,
        customdata: train_0.customdata,
        hovertemplate: hover_template,
        name: '無心臟病 (訓練)',
        marker: { color: 'blue', symbol: 'circle', size: 8, line: { color: 'black', width: 1 } }
    };

    // 圖層 3：訓練 - 有心臟病 (紅色 O)
    const traceTrain1 = {
        type: 'scatter', mode: 'markers',
        x: train_1.x, y: train_1.y,
        customdata: train_1.customdata,
        hovertemplate: hover_template,
        name: '有心臟病 (訓練)',
        marker: { color: 'red', symbol: 'circle', size: 8, line: { color: 'black', width: 1 } }
    };

    // 圖層 4：測試 - 無心臟病 (藍色 ▲)
    const traceTest0 = {
        type: 'scatter', mode: 'markers',
        x: test_0.x, y: test_0.y,
        customdata: test_0.customdata,
        hovertemplate: hover_template,
        name: '無心臟病 (測試)',
        marker: { color: 'blue', symbol: 'triangle-up', size: 10, line: { color: 'black', width: 1 } }
    };

    // 圖層 5：測試 - 有心臟病 (紅色 ▲)
    const traceTest1 = {
        type: 'scatter', mode: 'markers',
        x: test_1.x, y: test_1.y,
        customdata: test_1.customdata,
        hovertemplate: hover_template,
        name: '有心臟病 (測試)',
        marker: { color: 'red', symbol: 'triangle-up', size: 10, line: { color: 'black', width: 1 } }
    };

    // 組合圖層 (不變)
    const plotData = [
        boundaryTrace,
        traceTrain0,
        traceTrain1,
        traceTest0,
        traceTest1
    ];

    // 版面配置 (Layout)
    const layout = {
        title: ` ${description.x1_feature} 、 ${description.x2_feature}`,
        xaxis: {
            title: description.x1_feature,
            zeroline: false
        },
        yaxis: {
            title: description.x2_feature,
            zeroline: false
        },
        hovermode: 'closest',
        legend: {
            traceorder: 'normal',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E2E2',
            borderwidth: 1
        },
        margin: { t: 80, b: 60, l: 60, r: 30 }
    };

    // 繪製圖表 (不變)
    Plotly.newPlot(chartDiv, plotData, layout, {
        responsive: true,
        displayModeBar: false // 隱藏工作列
    });

    chartDiv.classList.remove('loading');
}