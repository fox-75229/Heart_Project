/**
 * 導航選單在手機和桌面版本的功能實現
 * 當 DOM 內容載入後初始化
 * 
 * 功能特點：
 * - 點擊漢堡選單按鈕時切換手機版導航選單
 * - 當視窗尺寸調整至桌面版（>800px）時自動關閉手機選單
 * - 處理導航切換的無障礙屬性
 * 
 * 相依元素：
 * - 需要 '.nav-toggle' 元素（漢堡選單按鈕）
 * - 需要 '.nav-list' 元素（導航選單）
 * - 需要 'open' 類別來控制選單顯示狀態
 * 
 * @listens DOMContentLoaded - 初始化導航功能
 * @listens click - 切換手機選單
 * @listens resize - 處理響應式行為
 */

document.addEventListener('DOMContentLoaded', function () {
    const toggle = document.querySelector('.nav-toggle');
    const navList = document.querySelector('.nav-list');

    if (!toggle || !navList) return;

    toggle.addEventListener('click', function () {
        navList.classList.toggle('open');
    });

    // 在視窗尺寸改變到桌機時，自動關閉手機選單以避免遺留狀態
    window.addEventListener('resize', function () {
        // console.log(window.innerWidth)
        if (window.innerWidth > 768) {
            if (navList.classList.contains('open')) {
                navList.classList.remove('open');
            }
            /*if (toggle.classList.contains('open')) {
                toggle.classList.remove('open');
                toggle.setAttribute('aria-expanded', 'false');
            }*/
        }
    });
});