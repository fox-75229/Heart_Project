/**
 * decision_tree.js
 *
 * 功能：
 * 1. 載入決策樹模型指標 (loadData)
 * 2. 載入 Graphviz SVG 圖表 (loadGraphviz)
 * 3. 綁定互動控制事件 (setupEventListeners)
 * 4. 處理預測請求 (handlePrediction)
 * 5. 更新預測文字顯示 (updatePredictionText)
 * 6. 高亮 SVG 樹狀圖節點 (highlightLeafNode)
 * 7. 注入 SVG CSS 樣式 (injectGraphvizCSS)
 */

document.addEventListener("DOMContentLoaded", () => {
    loadData();           // 載入模型指標與描述
    loadGraphviz();       // 載入決策樹 SVG
    setupEventListeners(); // 綁定滑桿、Radio 與按鈕事件
});

// -----------------------------------------------------
// 載入模型指標與描述
// -----------------------------------------------------
async function loadData() {
    try {
        const response = await fetch('/api/decision_tree/data');
        if (!response.ok) throw new Error(`HTTP 錯誤: ${response.status}`);

        const result = await response.json();
        if (result.success) {
            updateMetrics(result.metrics);          // 更新指標
            updateModelInfo(result.description);    // 更新資料集描述
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error("載入決策樹資料失敗:", error);
    }
}

// -----------------------------------------------------
// 載入 Graphviz SVG 樹狀圖
// -----------------------------------------------------
async function loadGraphviz() {
    const chartDiv = document.getElementById('graphviz-chart');
    try {
        const response = await fetch('/api/decision_tree/graph');
        if (!response.ok) throw new Error('載入 SVG 失敗');

        const svgData = await response.text();
        chartDiv.innerHTML = svgData;
        chartDiv.classList.remove('loading');

        injectGraphvizCSS(); // 注入 CSS 樣式
    } catch (error) {
        chartDiv.innerHTML = `<p class="error-text">❌ 載入樹狀圖失敗: ${error.message}</p>`;
    }
}

// -----------------------------------------------------
// 綁定互動控制事件
// -----------------------------------------------------
function setupEventListeners() {
    // ST 段斜率滑桿
    const slider = document.getElementById('st-slider');
    if (slider) {
        slider.addEventListener('input', () => {
            clearPrediction();
        });
    }

    // 運動心絞痛選項
    const radios = document.querySelectorAll('input[name="angina_group"]');
    radios.forEach(radio => radio.addEventListener('change', clearPrediction));

    // 預測按鈕
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }
}

// -----------------------------------------------------
// 處理預測請求
// -------------------------------------------  ----------
async function handlePrediction() {
    const predictBtn = document.getElementById('predict-btn');
    const loadingSpinner = document.getElementById('predict-loading');

    clearPrediction();
    predictBtn.disabled = true;
    loadingSpinner.style.display = 'block';

    try {
        const stSlope = document.getElementById('st-slider').value;
        const angina = document.querySelector('input[name="angina_group"]:checked').value;

        const response = await fetch('/api/decision_tree/predict', {
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
            const riskClass = updatePredictionText(stSlope, angina);
            highlightLeafNode(result.leaf_id, riskClass);
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

// -----------------------------------------------------
// 更新預測文字顯示
// -----------------------------------------------------
function updatePredictionText(stSlope, angina) {
    const container = document.getElementById('predict-container');
    const textElement = container.querySelector('.predict-text p');

    let riskLevel = "";
    let riskClass = "";
    let suggestionText = "";

    if (stSlope == 0 && angina == 0) {
        riskLevel = "中"; riskClass = "risk-medium";
        suggestionText = "可能有無症狀心肌缺血的風險 建議定期追蹤";
    } else if (stSlope == 0 && angina == 1) {
        riskLevel = "高"; riskClass = "risk-high";
        suggestionText = "患有心臟病風險極高 建議即刻安排儀器進行檢查";
    } else if (stSlope == 1 && angina == 0) {
        riskLevel = "中"; riskClass = "risk-medium";
        suggestionText = "可能有無症狀心肌缺血的風險 若患有糖尿病或高齡個體 建議定期追蹤";
    } else if (stSlope == 1 && angina == 1) {
        riskLevel = "高"; riskClass = "risk-high";
        suggestionText = "患有心臟病風險極高 建議即刻安排儀器進行檢查";
    } else if (stSlope == 2 && angina == 0) {
        riskLevel = "低"; riskClass = "risk-low";
        suggestionText = "健康個體，身體狀況良好";
    } else if (stSlope == 2 && angina == 1) {
        riskLevel = "中"; riskClass = "risk-medium";
        suggestionText = "身體狀況良好 痛感或許來自肌肉拉傷";
    } else {
        riskLevel = "未知"; riskClass = "risk-medium";
        suggestionText = "無法判斷，請確認輸入。";
    }

    textElement.innerHTML = `
        經過預測:本案例屬於:
        <strong id="predict-risk-level" class="deg ${riskClass}">${riskLevel}</strong>
        風險族群
        <strong id="predict-suggestion" class="sug">${suggestionText}</strong>
    `;
    container.style.display = 'block';

    return riskClass;
}

// -----------------------------------------------------
// 高亮 SVG 樹狀圖節點
// -----------------------------------------------------
function highlightLeafNode(leafId, riskClass) {
    const svg = document.getElementById('graphviz-chart').querySelector('svg');
    if (!svg) return;

    const targetInternalId = String(leafId);
    const allNodes = svg.querySelectorAll('.node');

    allNodes.forEach(node => {
        const titleEl = node.querySelector('title');
        if (titleEl && titleEl.textContent === targetInternalId) {
            node.classList.add(riskClass);
        }
    });
}

// -----------------------------------------------------
// 注入 SVG CSS 樣式
// -----------------------------------------------------
function injectGraphvizCSS() {
    const svg = document.getElementById('graphviz-chart').querySelector('svg');
    if (!svg) return;

    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
    style.textContent = `
        /* 1. (不變) 節點 hover 效果 */
        .node { cursor: pointer; transition: transform 0.2s ease, filter 0.2s ease; transform-origin: center; }
        .node:hover { transform: scale(1.02); filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3)); }

        /* 2. 【高】風險 (紅色) */
        .node.risk-high { filter: drop-shadow(0 0 8px #FF4136); }
        .node.risk-high polygon { stroke: #FF4136; stroke-width: 8px; }
        /* (我們目前只高亮葉節點，所以不需要 .edge 規則) */

        /* 3. 【中】風險 (橘色) */
        .node.risk-medium { filter: drop-shadow(0 0 8px #FFDC00); }
        .node.risk-medium polygon { stroke: #FFDC00; stroke-width: 8px; }

        /* 4. 【低】風險 (綠色) */
        .node.risk-low { filter: drop-shadow(0 0 8px #01FF70); }
        .node.risk-low polygon { stroke: #01FF70; stroke-width: 8px; }
    `;
    svg.prepend(style);
}

// -----------------------------------------------------
// 清除舊的預測與高亮
// -----------------------------------------------------
function clearPrediction() {
    document.getElementById('predict-container').style.display = 'none';
    const svg = document.getElementById('graphviz-chart').querySelector('svg');
    if (svg) {
        const highlightedNodes = svg.querySelectorAll('.risk-high, .risk-medium, .risk-low');
        highlightedNodes.forEach(el => {
            el.classList.remove('risk-high', 'risk-medium', 'risk-low');
        });
    }
}

// -----------------------------------------------------
// 更新模型指標與描述
// -----------------------------------------------------
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
