document.addEventListener("DOMContentLoaded", function () {
    // select-picture クラス内の画像を取得
    const imageElement = document.querySelector(".select-picture img");

    // 画像が読み込まれた後に幅をチェック
    if (imageElement) {
        imageElement.addEventListener("load", function () {
            if (imageElement.naturalWidth > 400 || imageElement.naturalHeight > 400) {
                // スタイルを無効にする
                imageElement.style.width = '';
                imageElement.style.height = '';
            } else {
                imageElement.style.width = imageElement.naturalWidth + 'px';
                imageElement.style.height = imageElement.naturalHeight + 'px';

                

            }
        });

        // 既に画像が読み込まれている場合の対処
        if (imageElement.complete) {
            imageElement.dispatchEvent(new Event("load"));
        }
    }
});
