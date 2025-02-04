const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';

//Variable global para almacenar el modelo
let movenet = undefined;

//Guardamos la img que tenemos en el .html
const EXAMPLE_IMG = document.getElementById('exampleImg');

//Obtenemos el canvas del html
const CROPPED_CANVAS = document.getElementById('croppedCanvas');

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
    // ** OJO: en el resultado nos da primero la altura y luego el ancho, en este caso es [360, 640, 3]
    console.log('Forma del tensor de la img: ', imageTensor.shape);

    //punto de incio para recortar la img [y, x, canal]
    // **El canal se deja en 0 porque queremos q el recorte cuente con los 3 canales de RGB
    let cropStartPoint = [15, 170, 0];

    //Establecemos el tamaño del recorte del tensor [alto, ancho, canales]
    let cropSize = [345, 345, 3];

    //Recorta el tensor de la img desde el punto de incio y con el tamaño q antes hemos definido
    let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);
    console.log('Tamaño del tensor recortado: ', croppedTensor.shape);

    //Mostramos la imagen recortada en el HTMLElement
    await tf.browser.toPixels(croppedTensor, CROPPED_CANVAS);

    //Redimensiona el tensor de la img recortado a 192x192 utilizando la interpolación bilineal
    // **Utilizamos el 3º parametro en true para mantener el aspect-ratio
    let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt();
    //Mostramos la nueva forma del tensor redimensionado q es [192, 192, 3]
    console.log('Tamaño del tensor ajustado: ', resizedTensor.shape);

    //Actualmente el tensor tiene 3 dimensiones y debemos expandirlo para q tenga 4 (es lo q espera en el input el modelo)
    // Asi que pasa a ser [1, 192, 192, 3] y le pasamos el tensor de entrada al modelo para q nos de una predicción
    let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));

    //convertimos el tensor de salida a un array de JS para q sea + fácil su uso (= a menos bugs)
    let arrayOutput = await tensorOutput.array();
    console.log('Predicciones:', arrayOutput);

    //Verificamos si la salida tiene keypoints
    let keypoints = arrayOutput[0][0];
    console.log('arrayOutput: ',arrayOutput[0][0]);

    //Pasamos los valores necesarios a la funcion para dibujar los keypoints + el valor ajustado del cropSize al tamaño del tensor / img
    //drawKeypoints(keypoints, CROPPED_CANVAS, cropSize[0] / 192);
    drawSkeleton(keypoints, CROPPED_CANVAS, cropSize[0] / 192);
}

//Ejecutamos la función
loadAndRunModel();


function drawSkeleton(keypoints, canvas, scale) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "yellow";
    ctx.lineWidth = 2;

    //Creamos el array con los puntos a conectar
    const connections = [
        [0, 1], [0, 2], //Nariz a ojos
        [1, 3], [2, 4], //Ojos a orejas
        [5, 7], [6, 8], //Hombros a codos
        [7, 9], [8, 10], //Codos a muñecas
        [11, 13], [12, 14], //Caderas a rodillas
        [13, 15], [14, 16], //Rodillas a tobillos
        [5, 11], [6, 12] //Hombros a caderas
    ];

    //Dibujamos las lineas del esqueleto
    connections.forEach(([i, j]) => {
        let x1 = keypoints[i][1] * 192 * scale; //Sacamos la x de su posición en el array y lo reescalamos al formato esperado
        let y1 = keypoints[i][0] * 192 * scale;
        let x2 = keypoints[j][1] * 192 * scale;
        let y2 = keypoints[j][0] * 192 * scale;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    });

    //Para dibujar los circulos de los keypoints
    keypoints.forEach(point => {
        let x = point[1] * 192 * scale;
        let y = point[0] * 192 * scale;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI); //Con los valores dibujamos el ciruclo
        ctx.fill();
    });

    //Limpiamos / eliminamos los tensores al finalizar para evitar fugas de memoria
    exampleInputTensor.dispose();
    imageTensor.dispose();
    croppedTensor.dispose();
    resizedTensor.dispose();
    tensorOutput.dispose();

}

/*
    Indice de datos
	0 - Nariz
	1 - Ojo izquierdo
	2 - Ojo derecho
	3 - Oreja izquierda
	4 - Oreja derecha
	5 - Hombro izquierdo
	6 - Hombro derecho
	7 - Codo izquierdo
	8 - Codo derecho
    9 - Muñeca izquierda
    10 - Muñeca derecha
    11 - Cadera izquierda
    12 - Cadera derecha
    13 - Rodilla izquierda
    14 - Rodilla derecha
    15 - Tobillo izquierdo
    16 - Tobillo derecho
 */