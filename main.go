package main

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"net/http"
	"reflect"
	"runtime/debug"

	"github.com/gin-gonic/gin"
	"github.com/nfnt/resize"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	tfgo "github.com/galeone/tfgo"
)

func classifyHandler(c *gin.Context) {
	imageSize := 224
	// Load image from form data
	file, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No image uploaded"})
		return
	}
	src, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open uploaded file"})
		return
	}
	defer src.Close()

	// Read image data and preprocess
	img, err := png.Decode(src)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to decode image"})
		return
	}

	// // preprocess the image
	// img = imaging.Resize(img, 224, 224, imaging.Lanczos)
	resized := resize.Resize(uint(imageSize), uint(imageSize), img, resize.MitchellNetravali)

	// Center crop the image
	bounds := resized.Bounds()
	x := (bounds.Max.X - imageSize) / 2
	y := (bounds.Max.Y - imageSize) / 2
	cropped := resized.(interface {
		SubImage(r image.Rectangle) image.Image
	}).SubImage(image.Rect(x, y, x+imageSize, y+imageSize))

	// Normalize the image pixel values
	var tfImage [1][][][3]float32
	mean := []float32{0.485, 0.456, 0.406}
	stddev := []float32{0.229, 0.224, 0.225}
	for j := 0; j < imageSize; j++ {
		tfImage[0] = append(tfImage[0], make([][3]float32, imageSize))
	}
	for i := 0; i < imageSize; i++ {
		for j := 0; j < imageSize; j++ {
			r, g, b, _ := cropped.At(j, i).RGBA()
			tfImage[0][i][j][0] = (float32(r)/65535.0 - mean[0]) / stddev[0]
			tfImage[0][i][j][1] = (float32(g)/65535.0 - mean[1]) / stddev[1]
			tfImage[0][i][j][2] = (float32(b)/65535.0 - mean[2]) / stddev[2]
		}
	}

	// Create a tensor from the preprocessed input
	imageTensor, _ := tf.NewTensor(tfImage)

	imgTf, err := imageToTensor(img, 224, 224)
	fmt.Println("==============================================================================================================================")
	fmt.Println(imgTf)
	fmt.Println(imageTensor)
	// fmt.Println(inputVals)

	// imageTensor, err := tensorFromImage

	// Load model
	model := tfgo.LoadModel("saved_model", []string{"serve"}, nil)

	output := model.Exec([]tf.Output{model.Op("StatefulPartitionedCall", 0)}, map[tf.Output]*tf.Tensor{
		model.Op("serving_default_resnet50_input", 0): imageTensor,
	})

	probabilities := output[0].Value().([][]float32)[0]

	softmax_classify := softmax(probabilities)
	sigmoid_classify := sigmoid(probabilities)
	reluClassify := relu(probabilities)
	tanhClassify := tanh(probabilities)
	fmt.Println(reflect.TypeOf(output[0].Value()))
	fmt.Println(output[0].Value())
	// Print the result
	fmt.Printf("Result: %v\n", probabilities)
	fmt.Println("softmax: ", softmax_classify)
	fmt.Println("sigmoid: ", sigmoid_classify)
	fmt.Println("ReLu: ", reluClassify)
	fmt.Println("Tanh: ", tanhClassify)

	fmt.Println(imageTensor)
	if softmax_classify[1] < 0.2694 {
		// if softmax_classify[1] < 0.271 {
		c.JSON(http.StatusOK, gin.H{"result": "kolam"})
	} else {
		c.JSON(http.StatusOK, gin.H{"result": "bukan kolam"})
	}

	// c.JSON(http.StatusOK, gin.H{"result": "image sent successfully"})
}

func imageToTensor(img image.Image, imageHeight, imageWidth int) (tfTensor *tf.Tensor, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("classify: %s (panic)\nstack: %s", r, debug.Stack())
		}
	}()

	if imageHeight <= 0 || imageWidth <= 0 {
		return tfTensor, fmt.Errorf("classify: image width and height must be > 0")
	}

	var tfImage [1][][][3]float32

	for j := 0; j < imageHeight; j++ {
		tfImage[0] = append(tfImage[0], make([][3]float32, imageWidth))
	}

	for i := 0; i < imageWidth; i++ {
		for j := 0; j < imageHeight; j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			tfImage[0][j][i][0] = convertValue(r)
			tfImage[0][j][i][1] = convertValue(g)
			tfImage[0][j][i][2] = convertValue(b)
		}
	}
	return tf.NewTensor(tfImage)
}

func convertValue(value uint32) float32 {
	return (float32(value >> 8)) / float32(255)
}

func softmax(input []float32) []float32 {
	sum := float32(0)
	output := make([]float32, len(input))
	for i := range input {
		output[i] = float32(math.Exp(float64(input[i])))
		sum += output[i]
	}
	for i := range output {
		output[i] /= sum
	}
	return output
}

func sigmoid(input []float32) []float32 {
	var predictions []float32
	// Apply sigmoid to the predictions
	for _, rawPrediction := range input {
		prediction := 1.0 / (1.0 + math.Exp(float64(-rawPrediction)))
		predictions = append(predictions, float32(prediction))
	}

	return predictions
}

func relu(input []float32) []float32 {
	for i := range input {
		if input[i] < 0 {
			input[i] = 0
		}
	}
	return input
}

func tanh(input []float32) []float32 {
	output := make([]float32, len(input))
	for i, val := range input {
		output[i] = float32(math.Tanh(float64(val)))
	}
	return output
}

func main() {
	r := gin.Default()

	// Set up classification endpoint
	r.POST("/classify", classifyHandler)

	// Start server
	if err := r.Run(":8080"); err != nil {
		fmt.Println(err)
	}
}
