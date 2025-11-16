
/**
 * static/js/decision_tree.js
 * (v2 - 互動預測版)
 *
 * 功能:
 * 1. 載入模型指標 (loadData)
 * 2. 載入 Graphviz SVG (loadGraphviz)
 * 3. 綁定「預測」按鈕事件 (setupEventListeners)
 * 4. 處理預測 (handlePrediction)
 * 5. 更新預測文字 (updatePredictionText)
 * 6. 高亮 SVG 路徑 (highlightTreePath)
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. 載入頁面基礎資料 (指標 + 模型資訊)
    loadData();

    // 2. 載入 SVG 樹狀圖
    loadGraphviz();

    // 3. 綁定互動控制項
    setupEventListeners();
});

// -----------------------------------------------------
// 1. 載入基礎資料 (指標、模型資訊)
// -----------------------------------------------------
async function loadData() {
    try {
        const response = await fetch('/api/decision_tree/data');
        if (!response.ok) {
            throw new Error(`HTTP 錯誤! 狀態: ${response.status}`);
        }
        const result = await response.json();

        if (result.success) {
            updateMetrics(result.metrics);
            updateModelInfo(result.description);
        } else {
            throw new Error(result.error);
        }
    } catch (error) {
        console.error("載入決策樹資料失敗:", error);
        // 你可以在 .metrics-container 顯示錯誤
    }
}

// -----------------------------------------------------
// 2. 載入 Graphviz SVG 圖表
// -----------------------------------------------------
async function loadGraphviz() {
    const chartDiv = document.getElementById('graphviz-chart');
    const loadingSpinner = chartDiv.querySelector('.loading-spinner');
    const loadingText = chartDiv.querySelector('.loading-text');

    try {
        const response = await fetch('/api/decision_tree/graph');
        if (!response.ok) throw new Error('載入 SVG 失敗');

        const svgData = await response.text();
        chartDiv.innerHTML = svgData;

        // 移除載入提示
        if (loadingSpinner) loadingSpinner.remove();
        if (loadingText) loadingText.remove();
        chartDiv.classList.remove('loading');

        // 【關鍵】注入 CSS 樣式 (包含高亮)
        injectGraphvizCSS();

    } catch (error) {
        chartDiv.innerHTML = `<p class="error-text">❌ 載入樹狀圖失敗: ${error.message}</p>`;
    }
}

// -----------------------------------------------------
// 3. 綁定所有事件監聽
// -----------------------------------------------------
function setupEventListeners() {
    // A. ST 段斜率滑桿
    const slider = document.getElementById('st-slider');//==================================================================================
    const stValue = document.getElementById('st-value'); //==================================================================================
    if (slider) {
        slider.addEventListener('input', () => {
            stValue.textContent = slider.value;
            // (可選) 滑動時清除舊的預測
            clearPrediction();
        });
    }

    // B. 運動心絞痛 Radio
    const radios = document.querySelectorAll('input[name="angina_group"]');
    radios.forEach(radio => {
        radio.addEventListener('change', () => {
            // (可選) 變更時清除舊的預測
            clearPrediction();
        });
    });

    // C. 預測按鈕
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }
}

// -----------------------------------------------------
// 4. 【核心】處理預測
// -----------------------------------------------------
async function handlePrediction() {
    const predictBtn = document.getElementById('predict-btn');
    const loadingSpinner = document.getElementById('predict-loading');

    // (v5 版的清除函式 - 它會清除舊高亮和隱藏舊文字)
    clearPrediction();

    // 顯示載入中
    predictBtn.disabled = true;
    loadingSpinner.style.display = 'block';

    try {
        // A. 獲取輸入值 (我們需要保留這兩個變數)
        const stSlope = document.getElementById('st-slider').value;//=======================================================================
        const angina = document.querySelector('input[name="angina_group"]:checked').value;

        // B. 呼叫預測 API
        const response = await fetch('/api/decision_tree/predict', {
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
            updatePredictionText(stSlope, angina);

        } else {
            throw new Error(result.error);
        }

    } catch (error) {
        console.error("預測時發生錯誤:", error);
        // 顯示錯誤 (例如在 predict-container)
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
// 5. 更新預測文字 (高/中/低 風險)
// -----------------------------------------------------
function updatePredictionText(stSlope, angina) {
    const container = document.getElementById('predict-container');
    const textElement = container.querySelector('.predict-text p');

    let riskLevel = "";
    let suggestionText = "";
    let riskClass = "";

    // --- 【【【關鍵修改：使用 6 種輸入組合】】】 ---

    // (stSlope == 0, angina == 0)
    if (stSlope == 0 && angina == 0) {
        riskLevel = "中";
        riskClass = "risk-medium";
        suggestionText = "可能有無症狀心肌缺血的風險 建議定期追蹤";
    }
    // (stSlope == 0, angina == 1)
    else if (stSlope == 0 && angina == 1) {
        riskLevel = "高";
        riskClass = "risk-high";
        suggestionText = "患有心臟病風險極高 建議即刻安排儀器進行檢查";
    }
    // (stSlope == 1, angina == 0)
    else if (stSlope == 1 && angina == 0) {
        riskLevel = "中";
        riskClass = "risk-medium";
        suggestionText = "可能有無症狀心肌缺血的風險 若患有糖尿病或高齡個體 建議定期追蹤";
    }
    // (stSlope == 1, angina == 1)
    else if (stSlope == 1 && angina == 1) {
        riskLevel = "高";
        riskClass = "risk-high";
        suggestionText = "患有心臟病風險極高 建議即刻安排儀器進行檢查";
    }
    // (stSlope == 2, angina == 0)
    else if (stSlope == 2 && angina == 0) {
        riskLevel = "低";
        riskClass = "risk-low";
        suggestionText = "健康個體，身體狀況良好";
    }
    // (stSlope == 2, angina == 1)
    else if (stSlope == 2 && angina == 1) {
        riskLevel = "中";
        riskClass = "risk-medium";
        suggestionText = "身體狀況良好 痛感或許來自肌肉拉傷";
    }
    // 備用 (理論上不會觸發)
    else {
        riskLevel = "未知";
        riskClass = "risk-medium";
        suggestionText = "無法判斷，請確認輸入。";
    }
    // --- 【【【修改結束】】】 ---

    // 將 6 種結果之一注入 HTML
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
// 6. 高亮 SVG 決策路徑
// -----------------------------------------------------
function highlightLeafNode(leafId) {
    const chartDiv = document.getElementById('graphviz-chart');
    const svg = chartDiv.querySelector('svg');
    if (!svg) return;

    // (清除函式 clearPrediction 已經在 handlePrediction 中被呼叫過了)

    // 1. 將 API 回傳的 ID (數字) 轉為字串，以便比對
    const targetInternalId = String(leafId);

    // 2. 取得 SVG 中的「所有」節點
    const allNodes = svg.querySelectorAll('.node');

    // 3. 遍歷所有節點
    allNodes.forEach(node => {
        // 4. 找到該節點內部的 <title> 標籤
        const titleEl = node.querySelector('title');

        // 5. 如果 <title> 的「文字內容」等於我們的目標 ID
        if (titleEl && titleEl.textContent === targetInternalId) {

            // 這才是我們要高亮的正確節點！
            node.classList.add('highlight');

            // (例如：leafId = 9, 
            // 程式會找到 <g id="node10"> 
            // 因為它內部的 <title> 是 "9")
        }
    });
}

// -----------------------------------------------------
// 7. (輔助) 注入 CSS 樣式到 SVG
// -----------------------------------------------------
// -----------------------------------------------------
// 7. (輔助) 注入 CSS 樣式到 SVG (v3 - 修正版)
// -----------------------------------------------------
function injectGraphvizCSS() {
    const chartDiv = document.getElementById('graphviz-chart');
    const svg = chartDiv.querySelector('svg');
    if (!svg) return;

    // 建立 <style> 標籤
    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");

    // 【【【重點修改：將 filter 移到 .node.highlight】】】
    style.textContent = `
        /* A. 節點 (Node) 互動 (不變) */
        .node {
            cursor: pointer;
            transition: transform 0.2s ease, filter 0.2s ease;
            transform-origin: center;
        }
        .node:hover {
            transform: scale(1.02);
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
        }

        /* *
         * B. 【高亮】節點樣式 (修正)
         *
         * 【修改】 將 filter 套用在「整個 .node.highlight 群組」上
         * 這樣發亮效果會套用到群組內的所有元素 (框 + 文字)
         */
        .node.highlight {
            /* * 將 filter: drop-shadow 移到這裡！
             * 這會讓整個節點 (包含框和文字) 一起發亮
             */
            filter: drop-shadow(0 0 8px #ffc107);
        }
        
        /* 【修改】 這裡只改變 polygon (框) 的外框樣式 */
        .node.highlight polygon {
            stroke: #ffc107; /* 亮黃色外框 */
            stroke-width: 4px; 
            /* (已移除這裡的 filter) */
        }

        /* *
         * C. 【高亮】邊緣樣式 (修正)
         *
         * 【修改】 同理，將 filter 套在「.edge.highlight 群組」上
         */
        .edge.highlight {
            filter: drop-shadow(0 0 5px #ffc107); /* 讓線條和箭頭一起發亮 */
        }
        
        /* 【修改】 這裡只改變 path (線) 的外框樣式 */
        .edge.highlight path {
            stroke: #ffc107;
            stroke-width: 3px;
            /* (已移除這裡的 filter) */
        }
        
        /* D. 【高亮】邊緣樣式 (箭頭) (不變) */
        .edge.highlight polygon {
            fill: #ffc107;
            stroke: #ffc107;
        }
    `;
    svg.prepend(style); 
}


function clearPrediction() {

    document.getElementById('predict-container').style.display = 'none';


    const svg = document.getElementById('graphviz-chart').querySelector('svg');
    if (svg) {
        svg.querySelectorAll('.node.highlight, .edge.highlight')
            .forEach(el => el.classList.remove('highlight'));
    }
}



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