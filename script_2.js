
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

let detector;
let faces;
let model;
let main_model;

const map = new Map();

// const x = document.getElementById("click");
// x.addEventListener("click", RespondClick);


map.set(0, "happy");
map.set(1, "neutral");
map.set(2, "sad");
map.set(3, "surprise");


const setupCamera = () =>{
    navigator.mediaDevices.getUserMedia({
        video: {width: 600, height: 400},
        audio: false,
    }).then(stream => {
        video.srcObject = stream;
    });
};

async function inference(){
    tf.engine().startScope()
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    faces = await detector.estimateFaces(video, {flipHorizontal: false});
    // console.log(faces);

    var start = [faces[0].box.xMin, faces[0].box.yMin]
    var end = [faces[0].box.xMax, faces[0].box.yMax]
    var size = [faces[0].box.width, faces[0].box.height]
    
    var inputImage = await tf.browser.fromPixels(video)
    inputImage = inputImage.toFloat().div(tf.scalar(255))
    inputImage=inputImage.slice([parseInt(start[1]),parseInt(start[0]),0],[parseInt(size[1]),parseInt(size[0]),3])
    inputImage=inputImage.resizeBilinear([48, 48]).reshape([1, 48, 48, 3])
    var results = main_model.predict(inputImage).dataSync()
    results = Array.from(results)
    // console.log(results)
    text = map.get(results.indexOf(Math.max(...results)))
    ctx.drawImage(video, 0, 0, 600, 400);
    ctx.beginPath();

    ctx.lineWidth = "4";
    ctx.strokeStyle = "blue";
    ctx.rect(
        faces[0].box.xMin,
        faces[0].box.yMin,
        faces[0].box.height,
        faces[0].box.width
    );
    ctx.font = "bold 15pt sans-serif";
    ctx.fillText(text,start[0]+5,start[1]+20)
    ctx.stroke();
    tf.engine().endScope()
};


setupCamera();
video.addEventListener("loadeddata", async() =>{
    model = faceDetection.SupportedModels.MediaPipeFaceDetector;
    detector = await faceDetection.createDetector(model,   {runtime: 'mediapipe',
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection',});
    main_model = await tf.loadLayersModel('new_tfjs/model.json');
    setInterval(inference, 150);
});