/**
 * 首頁腳本 (index.js)
 * * 說明：
 * 此檔案負責 index.html 頁面的互動特效。
 * 主要功能：驅動 Hero 區塊的滑鼠聚光燈 (Spotlight) 特效。
 */
// 等待 DOM 內容完全載入後再執行
document.addEventListener("DOMContentLoaded", () => {
    //選取 Hero 區塊元素
    const hero = document.querySelector(".hero-section")

    //檢查是否成功選取
    if (hero) {
        //監聽滑鼠在 hero 區塊上的移動事件
        hero.addEventListener("mousemove", (e) => {
            //取得.hero-section位子
            const rect = hero.getBoundingClientRect();
            //計算滑鼠在裡面的x,y座標
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            //x,y座標寫入css觸發光源位置
            e.target.style.setProperty("--mouse-x", `${x}px`)
            e.target.style.setProperty("--mouse-y", `${y}px`)
        });
        //4. 監聽滑鼠移出 hero 區塊的事件
        hero.addEventListener("mouseleave", () => {
            // 滑鼠移出時，將聚光燈位置重設回預設的中間 (50% 50%)
            hero.style.setProperty("--mouse-x", `50%`);
            hero.style.setProperty("--mouse-y", `50%`);
        });
    }
})