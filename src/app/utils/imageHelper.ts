import * as Jimp from "jimp";
import { Tensor } from "onnxruntime-web";
// import { exec } from 'child_process';
import { promisify } from "util";

export async function getImageTensorFromPath(
  path: string,
  dims: number[] = [1, 3, 224, 224]
): Promise<Tensor> {
  // 1. load the image
  var image = await loadImageFromPath(path, dims[2], dims[3]);
  // 2. convert to tensor
  var imageTensor = imageDataToTensor(image, dims);
  // 3. return the tensor
  return imageTensor;
}

// const execAsync = promisify(exec);

async function loadImageFromPath(
  path: string,
  width: number = 224,
  height: number = 224
): Promise<Jimp> {
  // Define paths for intermediate and final images
  const dicomImagePath = path;
  const convertedImagePath = "./data/input.png";

  // Use DCMTK's dcmtopgm to convert DICOM to PNG
  // await execAsync(`dcmtopgm ${dicomImagePath} ${convertedImagePath}`);

  // Use Jimp to load the converted PNG image and resize it
  const imageData = await Jimp.read(convertedImagePath).then(
    (imageBuffer: Jimp) => {
      return imageBuffer.resize(width, height);
    }
  );

  return imageData;
}

async function processImage() {
  try {
    // Use Jimp to read the PNG image
    const image = await Jimp.read("path/to/your/output.png");

    // Manipulate the image (e.g., resize)
    const resizedImage = await image.resize(224, 224);

    // Save the manipulated image
    await resizedImage.writeAsync("path/to/your/output_resized.png");
  } catch (error) {
    console.error("Error:", error);
  }
}

function imageDataToTensor(image: Jimp, dims: number[]): Tensor {
  // 1. Get buffer data from image and create R, G, and B arrays.
  var imageBufferData = image.bitmap.data;
  const [redArray, greenArray, blueArray] = new Array(
    new Array<number>(),
    new Array<number>(),
    new Array<number>()
  );

  // 2. Loop through the image buffer and extract the R, G, and B channels
  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(imageBufferData[i]);
    greenArray.push(imageBufferData[i + 1]);
    blueArray.push(imageBufferData[i + 2]);
    // skip data[i + 3] to filter out the alpha channel
  }

  // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
  const transposedData = redArray.concat(greenArray).concat(blueArray);

  // 4. convert to float32
  let i,
    l = transposedData.length; // length, we need this for the loop
  // create the Float32Array size 3 * 224 * 224 for these dimensions output
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / 255.0; // convert to float
  }
  // 5. create the tensor object from onnxruntime-web.
  const inputTensor = new Tensor("float32", float32Data, dims);
  return inputTensor;
}
