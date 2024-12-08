// モーダルを表示する関数
function showModal(index) {
    var modal = document.getElementById('modal-' + index);  // モーダルのIDを取得
    modal.classList.add('show');  // モーダルを表示
  }
  
  // モーダルを閉じる関数
  function closeModal(index) {
    var modal = document.getElementById('modal-' + index);  // モーダルのIDを取得
    modal.classList.remove('show');  // モーダルを非表示にする
  }
  
  // モーダル外の部分をクリックしても閉じる
  window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
      var modals = document.getElementsByClassName('modal');
      for (var i = 0; i < modals.length; i++) {
        modals[i].classList.remove('show');
      }
    }
  };
  