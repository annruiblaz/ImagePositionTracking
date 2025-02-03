const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';

//Variable global para almacenar el modelo
let movenet = undefined;

//Guardamos la img que tenemos en el .html
const EXAMPLE_IMG = document.getElementById('exampleImg');

//Funcion asincrona para cargar y ejecutar el modelo
async function loadAndRunModel() {
    //cargamos el modelo de tensorFlow hub
    movenet = await tf.loadGraphModel (MODEL_PATH, {fromTFHub: true});
    /* creamos un tensor de entrada de dimensiones [1, 192, 192, 3]
        - 1: es el batch size (cuantas imagenes procesa a la vez, en este caso 1)
        - 192, 192: son las dimensiones d la img de entrada q espera el modelo
        - 3: RGB de la img

        Ademas el tipo de datos es int32 asi q solo admite valores enteros como input
    */
    let exampleInputTensor = tf.zeros([1, 192, 192, 3], 'int32');

    //convertimos la img para q sea compatible con tensorFlow
    let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
    //al printearlo comprobamos q no la podemos utilizar porque tiene una forma (height y width) distintas a la q espera el input
    //OJO: en el resultado nos da primero la altura y luego el ancho, en este caso es [360, 640, 3]
    console.log('Forma del tensor de la img: ', imageTensor.shape);

    //punto de incio para recortar la img [y, x, canal]
    // **El canal se deja en 0 porque queremos q el recorte cuente con los 3 canales de RGB
    let cropStartPoint = [15, 170, 0];

    //Establecemos el tamaño del recorte del tensor [alto, ancho, canales]
    let cropSize = [345, 345, 3];

    //Recorta el tensor de la img desde el punto de incio y con el tamaño q antes hemos definido
    let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);

    //Redimensiona el tensor de la img recortado a 192x192 utilizando la interpolación bilineal
    // **Utilizamos el 3º parametro en true para mantener el aspect-ratio
    let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt();

    //Mostramos la nueva forma del tensor redimensionado q es [192, 192, 3]
    console.log('Tamaño del tensor ajustado: ', resizedTensor);

    //pasamos el tensor de entrada al modelo para q nos de una predicción
    let tensorOutput = movenet.predict(exampleInputTensor);

    //convertimos el tensor de salida a un array de JS para q sea + fácil su uso (= a menos bugs)
    let arrayOutput = await tensorOutput.array();

    console.log(arrayOutput);
}

//Ejecutamos la función
loadAndRunModel();