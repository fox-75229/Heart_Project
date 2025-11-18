/**
 * 基礎腳本 (base.js)
 * * 處理 base.html 中的響應式導覽列（漢堡選單）的互動功能。
 * * 功能：
 * 1. 監聽漢堡選單按鈕 ('.nav-toggle') 的點擊事件。
 * 2. 切換導覽選單 ('.nav-list') 的 '.open' CSS class，以顯示或隱藏選單。
 * 3. 同步更新按鈕的 'aria-expanded' 屬性，這對於 CSS 動畫 (X 變形) 和無障礙訪問 (A11y) 至關重要。
 * 4. 監聽視窗 'resize' 事件，當視窗拉寬回桌機版時，自動關閉手機選單並重置按鈕狀態。
 */

// 等待 DOM 內容完全載入後再執行腳本
document.addEventListener('DOMContentLoaded', function () {

    // 選取必要的元素
    const toggleButton = document.querySelector('.nav-toggle'); // 漢堡選單按鈕
    const navList = document.querySelector('.nav-list');      // 導覽選單 UL

    // 如果頁面上找不到這兩個元素中的任何一個，則停止執行，避免錯誤
    if (!toggleButton || !navList) {
        console.warn("未找到導覽列元素 (nav-toggle 或 nav-list)，base.js 停止運作。");
        return;
    }

    // 監聽漢堡按鈕的點擊事件
    toggleButton.addEventListener('click', function () {
        // 切換 .nav-list 上的 'open' class
        // base.css 中 .nav-list.open { display: flex !important; }
        navList.classList.toggle('open');

        // --- 關鍵修正 ---
        // 檢查 'open' class 是否存在，並用 'true'/'false' (字串) 來更新 aria-expanded 屬性
        // 這會觸發 base.css 中的 .nav-toggle[aria-expanded="true"] ... 動畫 (三條線變 X)
        const isExpanded = navList.classList.contains('open');
        toggleButton.setAttribute('aria-expanded', isExpanded.toString());
    });

    // 監聽視窗大小改變事件
    window.addEventListener('resize', function () {

        // 參照 base.css 中的 RWD 斷點 (768px)
        // 如果視窗寬度大於 768px (桌機版)
        if (window.innerWidth > 768) {

            // 檢查手機選單是否是開啟狀態
            if (navList.classList.contains('open')) {
                // 如果是，則自動關閉選單
                navList.classList.remove('open');

                // --- 關鍵修正 ---
                // 同時也要重置漢堡按鈕的狀態，否則按鈕會停留在 'X'
                toggleButton.setAttribute('aria-expanded', 'false');
            }
        }
    });
});