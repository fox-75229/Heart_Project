/**
 * static/js/decision_tree.js
 * * (這是 logistic.js v8 的複製品, 只是 API 路徑不同)
 * 1. 呼叫 API (/api/decision_tree/data)
 * 2. 更新指標卡
 * 3. 繪製 Plotly 決策邊界圖 (Heatmap + Scatter)
 */

// 1. 頁面載入完成後，執行 loadData 與 loadGraphviz
// 這裡同時載入 Plotly 圖表與 Graphviz 樹狀圖

document.addEventListener("DOMContentLoaded", () => {
    loadData();      // 載入決策邊界圖與指標
    loadGraphviz();  // 載入 Graphviz 樹狀圖
});

// 2. 呼叫 API，取得圖表和指標資料
async function loadData() {
    const chartDiv = document.getElementById('plotly-chart');
    const loadingSpinner = chartDiv.querySelector('.loading-spinner');
    const loadingText = chartDiv.querySelector('.loading-text');
    try {
        const response = await fetch('/api/decision_tree/data');
        if (!response.ok) {
            throw new Error(`HTTP 錯誤! 狀態: ${response.status}`);
        }
        const result = await response.json();
        if (result.success) {
            updateMetrics(result.metrics); // 更新指標
            drawPlot(result.data, result.description); // 繪製 Plotly 決策邊界圖
            if (loadingSpinner) loadingSpinner.remove();
            if (loadingText) loadingText.remove();
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error("載入決策樹資料失敗:", error);
        chartDiv.innerHTML = `<p class="error-text">❌ 載入圖表失敗: ${error.message}</p>`;
    }
}

// 3. 將 API 回傳的指標填入 HTML
function updateMetrics(metrics) {
    document.getElementById('metrics-accuracy').textContent = metrics.accuracy;
    document.getElementById('metrics-precision').textContent = metrics.precision;
    document.getElementById('metrics-recall').textContent = metrics.recall;
    document.getElementById('metrics-f1').textContent = metrics.f1;
    document.getElementById('metrics-auc').textContent = metrics.auc;
}

// 4. 使用 Plotly 繪製決策邊界圖
function drawPlot(data, description) {
    const chartDiv = document.getElementById('plotly-chart');
    const JITTER_AMOUNT = 0.15;
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
    const boundaryTrace = {
        type: 'heatmap',
        x: data.decision_boundary.xx[0],
        y: data.decision_boundary.yy.map(row => row[0]),
        z: data.decision_boundary.Z,
        colorscale: [
            ['0.0', 'rgb(74, 110, 184)'],
            ['1.0', 'rgb(187, 85, 85)']
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
    const plotData = [boundaryTrace, traceTrain0, traceTrain1, traceTest0, traceTest1];
    const layout = {
        title: `Decision Tree 決策邊界<br>(使用 ${description.x1_feature} 和 ${description.x2_feature})`,
        xaxis: { title: description.x1_feature, zeroline: false },
        yaxis: { title: description.x2_feature, zeroline: false },
        hovermode: 'closest',
        legend: {
            traceorder: 'normal', bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E2E2', borderwidth: 1
        },
        margin: { t: 80, b: 60, l: 60, r: 30 }
    };
    const config = {
        responsive: true,
        scrollZoom: false,
        modeBarButtons: [
            ['zoom2d', 'autoScale2d']
        ]
    };
    Plotly.newPlot(chartDiv, plotData, layout, config);
    chartDiv.classList.remove('loading');
}

//=====================================================================
//=====================================================================
//=====================================================================
function injectGraphvizCSS() {
    const chartDiv = document.getElementById('graphviz-chart');
    const svg = chartDiv.querySelector('svg');
    if (svg) {
        // 建立 <style> 標籤
        const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
        style.textContent = `
            .node {
                cursor: pointer;
             transition: transform 0.25s ease, filter 0.25s ease;
                transform-origin: center;
            }
            .node:hover {
                transform: translateY(-0.5px) scale(1.005);
                filter: drop-shadow(0 2px 6px rgba(11, 52, 110, 0.5));
            }
        `;

        svg.appendChild(style);
        
    }
}
// 5. 載入 Graphviz SVG 樹狀圖
async function loadGraphviz() {
    const chartDiv = document.getElementById('graphviz-chart');
    chartDiv.innerHTML = '<span>載入中...</span>';
    try {
        const response = await fetch('/api/decision_tree/graph');
        if (!response.ok) throw new Error('載入失敗');
        const svg = await response.text();
        chartDiv.innerHTML = svg;
        injectGraphvizCSS(); // 新增這行
    } catch (error) {
        chartDiv.innerHTML = `<span style="color:red;">❌ 載入失敗: ${error.message}</span>`;
    }
}