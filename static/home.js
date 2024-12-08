document.addEventListener("DOMContentLoaded", function () {
    document.querySelector(".parent input[type=file]").addEventListener("change", function (event) {
        const file = event.target.files[0];
//        const previewText = document.querySelector(".file__none");
        const previewImg = document.querySelector(".preview-img");
        
        if (file) {
            const fileReader = new FileReader();
            fileReader.readAsDataURL(file);

            fileReader.onload = function (e) {
                const imgTag = document.createElement('img');
                imgTag.src = e.target.result;
                imgTag.alt = '選択された画像';

                // 画像を一度追加してから、サイズをチェック
                previewImg.innerHTML = '';
                previewImg.appendChild(imgTag);

                // 画像が読み込まれた後に幅と高さをチェック
                imgTag.onload = function () {
                    const imgWidth = imgTag.naturalWidth;
                    const imgHeight = imgTag.naturalHeight;

                    // 幅が423px以上または高さが432px以上の場合、styleを無効にする
                    if (imgWidth > 420 || imgHeight > 420) {
                        imgTag.style.width = ''; // スタイルをクリア
                        imgTag.style.height = ''; // スタイルをクリア
                    } else {
                        // それ以外の場合、スタイルを適用
                        imgTag.style.width = `${imgWidth}px`;
                        imgTag.style.height = `${imgHeight}px`;
                    }
                };

                submitBtn.disabled = false;
//                previewText.textContent = file.name;
            };

            fileReader.onerror = function () {
                previewText.textContent = "ファイルの読み込みに失敗しました";
            };
        } else {
            previewText.textContent = "選択されていません";
            previewImg.innerHTML = '';
            submitBtn.disabled = true;
        }
    });
});

function toggleHistory() {
    const historyDiv = document.getElementById('history');
    if (historyDiv.style.display === 'none') {
        historyDiv.style.display = 'block';
    } else {
        historyDiv.style.display = 'none';
    }
}

function clearFile(event) {
    event.preventDefault(); // デフォルトのフォーム送信を防ぐ

    // フォームのリセット
    const form = document.querySelector("form"); // 適切なフォームのセレクタを指定
    if (form) {
        form.reset(); // フォーム全体をリセット
    }

    // プレビュー画像とテキストのクリア
//  const previewText = document.querySelector(".file__none");
    const previewImg = document.querySelector(".preview-img");

    if (previewImg) {
        previewImg.innerHTML = ''; // プレビュー画像をクリア
    }

    submitBtn.disabled = true;
//    if (previewText) {
//        previewText.textContent = "選択されていません"; // テキストをクリア
//    }
}

window.onload = () => {
    const animatedElement = document.getElementById('animatedElement');
    animatedElement.classList.add('show');
    animatedElement.classList.add('fadeout', 'is-animated');

    // アニメーション終了後に pointer-events を無効化する
    animatedElement.addEventListener('animationend', () => {
        animatedElement.style.pointerEvents = 'none';
    });
};