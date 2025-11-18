/**
 * 決策樹頁面腳本 (decision_tree.js)
 *
 * 負責 decision_tree.html 頁面的所有互動功能。
 * 1. 頁面載入時，非同步獲取「模型指標」與「SVG 樹狀圖」。
 * 2. 綁定「滑桿」、「選項」和「預測按鈕」的事件。
 * 3. 呼叫預測 API，並在 SVG 上高亮顯示「決策路徑」。
 * 4. 更新預測結果與建議文字。
 */

// --- 1. 頁面初始化 ---

document.addEventListener("DOMContentLoaded", () => {
    // 立即執行兩個非同步載入任務
    loadData();        // 載入模型指標 (Accuracy, AUC...) 與描述
    loadGraphviz();    // 載入決策樹 SVG 圖表

    // 綁定所有互動元件的事件監聽
    setupEventListeners();
});

// --- 2. API 呼叫與資料載入 ---

/**
 * [非同步] 載入模型指標 (Metrics) 與詳細資訊 (Description)
 * 呼叫 API: /api/decision_tree/data
 */
async function loadData() {
    try {
        const response = await fetch('/api/decision_tree/data');
        if (!response.ok) throw new Error(`HTTP 錯誤: ${response.status}`);

        const result = await response.json();
        if (result.success) {
            updateMetrics(result.metrics);      // 將指標填入 UI
            updateModelInfo(result.description); // 將模型資訊填入 UI
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error("載入決策樹資料失敗:", error);
        // 在 UI 上顯示錯誤
    }
}

/**
 * [非同步] 載入 Graphviz 產生的 SVG 樹狀圖
 * 呼叫 API: /api/decision_tree/graph
 */
async function loadGraphviz() {
    const chartDiv = document.getElementById('graphviz-chart');
    try {
        const response = await fetch('/api/decision_tree/graph');
        if (!response.ok) throw new Error('載入 SVG 失敗');

        const svgData = await response.text(); // 獲取 SVG 的純文字內容
        chartDiv.innerHTML = svgData;         // 將 SVG 注入到 <div> 中
        chartDiv.classList.remove('loading'); // 移除載入動畫

        injectGraphvizCSS(); // 注入 CSS 樣式，使 SVG 可互動
    } catch (error) {
        chartDiv.innerHTML = `<p class="error-text">❌ 載入樹狀圖失敗: ${error.message}</p>`;
    }
}

/**
 * [非同步] 呼叫預測 API
 * 呼叫 API: /api/decision_tree/predict
 */
async function handlePrediction() {
    const predictBtn = document.getElementById('predict-btn');
    const loadingSpinner = document.getElementById('predict-loading');

    // --- 1. 準備 UI (進入載入狀態) ---
    clearPrediction(); // 清除上次的預測高亮
    predictBtn.disabled = true;
    loadingSpinner.style.display = 'block';

    try {
        // --- 2. 獲取使用者輸入 ---
        const stSlope = document.getElementById('st-slider').value;
        const angina = document.querySelector('input[name="angina_group"]:checked').value;

        // --- 3. 發送 POST 請求 ---
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

        // --- 4. 處理成功的回應 ---
        if (result.success) {
            const riskClass = updatePredictionText(stSlope, angina);
            // 根據 API 回傳的「葉節點 ID」高亮 SVG
            highlightLeafNode(result.leaf_id, riskClass);
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        // --- 5. 處理失敗 ---
        console.error("預測時發生錯誤:", error);
        const container = document.getElementById('predict-container');
        const textElement = container.querySelector('.predict-text p');
        if (textElement) {
            textElement.innerHTML = `<span class="error-text">❌ 預測失敗: ${error.message}</span>`;
        }
        container.style.display = 'block';
    } finally {
        // --- 6. 恢復 UI ---
        predictBtn.disabled = false;
        loadingSpinner.style.display = 'none';
    }
}

// --- 3. UI 更新函式 ---

/**
 * 更新「模型評估指標」區塊
 */
function updateMetrics(metrics) {
    document.getElementById('metrics-accuracy').textContent = metrics.accuracy;
    document.getElementById('metrics-precision').textContent = metrics.precision;
    document.getElementById('metrics-recall').textContent = metrics.recall;
    document.getElementById('metrics-f1').textContent = metrics.f1;
    document.getElementById('metrics-auc').textContent = metrics.auc;
}

/**
 * 更新「模型詳細資訊」區塊
 */
function updateModelInfo(description) {
    document.getElementById('desc-dataset').textContent = description.dataset;
    document.getElementById('desc-total-samples').textContent = description.total_samples;
    document.getElementById('desc-train-size').textContent = description.train_size;
    document.getElementById('desc-test-size').textContent = description.test_size;
    document.getElementById('desc-target').textContent = description.target;
    document.getElementById('desc-features').textContent = description.selected_features.join(', ');
}

/**
 * 根據 *使用者輸入* 更新預測結果的文字與建議
 */
function updatePredictionText(stSlope, angina) {
    const container = document.getElementById('predict-container');
    const textElement = container.querySelector('.predict-text p');

    let riskLevel = "";
    let riskClass = "";
    let suggestionText = "";

    // --- 這是前端重新實現的「決策樹邏輯」 ---
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
    } else{
        riskLevel = "中"; riskClass = "risk-medium";
        suggestionText = "身體狀況良好 痛感或許來自肌肉拉傷";
    } 
    // --- 邏輯結束 ---

    // 填入 HTML
    textElement.innerHTML = `
        經過預測:本案例屬於:
        <strong id="predict-risk-level" class="deg ${riskClass}">${riskLevel}</strong>
        風險族群
        <strong id="predict-suggestion" class="sug">${suggestionText}</strong>
    `;
    container.style.display = 'block';

    return riskClass; // 回傳 CSS class (risk-high, risk-medium, risk-low)
}


//  高亮 SVG 樹狀圖中的特定葉節點
//  @param {string} leafId - 從 API 獲取的葉節點 ID
//  @param {string} riskClass - 要添加的 CSS class (e.g., 'risk-high')
 
function highlightLeafNode(leafId, riskClass) {
    const svg = document.getElementById('graphviz-chart').querySelector('svg');
    if (!svg) return;

    const targetInternalId = String(leafId);

    // 尋找 SVG 中 <title> 標籤內容 = leafId 的節點
    // (優化建議：如果 Graphviz 產生的 <g> 標籤有 ID，用 getElementById 會更快)
    const allNodes = svg.querySelectorAll('.node');
    allNodes.forEach(node => {
        const titleEl = node.querySelector('title');
        // 找到節點 (g.node)
        if (titleEl && titleEl.textContent === targetInternalId) {
            node.classList.add(riskClass); // 添加高亮 class
        }
    });
}

/**
 * 將 CSS 樣式 (<style>) 注入到 SVG 內部
 * 這樣 CSS 才能控制 SVG 元素的 :hover 和 class 樣式
 */
function injectGraphvizCSS() {
    const svg = document.getElementById('graphviz-chart').querySelector('svg');
    if (!svg) return;

    // 建立一個 <style> 元素 (必須使用 SVG 的命名空間)
    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
    style.textContent = `
        /* 1. 節點 hover 效果 */
        .node { 
            cursor: pointer; 
            transition: transform 0.2s ease, filter 0.2s ease; 
            transform-origin: center; 
        }
        .node:hover { 
            transform: scale(1.02); 
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3)); 
        }

        /* 2. 【高】風險 (紅色) 高亮 */
        .node.risk-high { filter: drop-shadow(0 0 8px #FF4136); }
        .node.risk-high polygon { stroke: #FF4136; stroke-width: 8px; }

        /* 3. 【中】風險 (橘色) 高亮 */
        .node.risk-medium { filter: drop-shadow(0 0 8px #FFDC00); }
        .node.risk-medium polygon { stroke: #FFDC00; stroke-width: 8px; }

        /* 4. 【低】風險 (綠色) 高亮 */
        .node.risk-low { filter: drop-shadow(0 0 8px #01FF70); }
        .node.risk-low polygon { stroke: #01FF70; stroke-width: 8px; }
    `;
    svg.prepend(style); // 將 <style> 插入到 <svg> 的最頂部
}

/**
 * 清除舊的預測結果 (隱藏文字、移除 SVG 高亮)
 */
function clearPrediction() {
    document.getElementById('predict-container').style.display = 'none';
    const svg = document.getElementById('graphviz-chart').querySelector('svg');
    if (svg) {
        // 找到所有高亮的節點
        const highlightedNodes = svg.querySelectorAll('.risk-high, .risk-medium, .risk-low');
        // 移除它們的 class
        highlightedNodes.forEach(el => {
            el.classList.remove('risk-high', 'risk-medium', 'risk-low');
        });
    }
}

// --- 4. 事件綁定 ---

/**
 * 綁定所有互動元件的事件監聽器
 */
function setupEventListeners() {
    const slider = document.getElementById('st-slider');
    const predictBtn = document.getElementById('predict-btn');
    const radios = document.querySelectorAll('input[name="angina_group"]');

    // 1. 當滑桿或選項變更時，清除舊的預測結果
    if (slider) {
        slider.addEventListener('input', clearPrediction);
    }
    radios.forEach(radio => radio.addEventListener('change', clearPrediction));

    // 2. 點擊「執行預測」按鈕
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }
}