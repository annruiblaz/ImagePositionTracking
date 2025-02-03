const status = document.getElementById('status');

if (status) {
    status.innerText = 'TensorFlow.js Cargado - Version: ' + tf.version.tfjs;
}

//tf.loadGraphModel("/kaggle/input/movenet/tfjs/singlepose-lightning/1", { fromTFHub: true });
