console.log("Hello index")

document.addEventListener("DOMContentLoaded", () => {
    //選取css .hero-section元素
    const hero = document.querySelector(".hero-section")
    // console.log("找到的.hero-section元素",hero)

    //檢查是否選到
    if (hero) {
        //監聽滑鼠移動事件(mousemove)
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
        //
        hero.addEventListener("mouseleave", () => {
            hero.style.setProperty("--mouse-x", `50%`);
            hero.style.setProperty("--mouse-y", `50%`);
        });
    }
})