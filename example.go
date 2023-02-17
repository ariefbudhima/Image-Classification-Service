package main

import (
	// "bytes"
	"fmt"
	"image/png"
	"net/http"
	"image"

	"github.com/gin-gonic/gin"
	// "github.com/disintegration/imaging"
	"github.com/nfnt/resize"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
)

func classifyHandler(c *gin.Context) {
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
	// img = imaging.Resize(img, 224, 224, imaging.Lanczos)
	tensorInput, err := tf.NewTensor(preprocess(img))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create input tensor"})
		return
	}

	// Load model
	model := tg.LoadModel("saved_model", []string{"serve"}, nil)

	// Create a new graph to calculate the softmax output
	g := tf.NewGraph()
	softmaxInput := g.Placeholder(tf.Float, tf.MakeShape(1, 2))
	softmax := g.Softmax(softmaxInput, -1)

	// Execute the model and get the output tensor
	outputTensor := model.Output("dense_2/Softmax")[0]

	// Run the session to get the classification result
	session, err := tf.NewSession(model.Graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("serving_default_input_1").Output(0): tensorInput,
		},
		[]tf.Output{
			outputTensor,
		},
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}
	
	// Get the softmax probabilities
	softmaxProbabilities, err := session.Run(
		map[tf.Output]*tf.Tensor{
			softmaxInput: outputTensor,
		},
		[]tf.Output{
			softmax,
		},
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}
	
	// Convert the output tensor to a float slice
	outputData := output[0].Value().([][]float32)[0]
	
	// Get the classification result based on the argmax of the output
	if outputData[0] > outputData[1] {
		fmt.Println("Not pond")
	} else {
		fmt.Println("Pond")
	}
	
	// Get the softmax probabilities for the classification result
	softmaxProb := softmaxProbabilities[0].Value().([][]float32)[0]
	fmt.Printf("Pond probability: %f\n", softmaxProb[1])
	fmt.Printf("Not pond probability: %f\n", softmaxProb[0])

	// // Create input tensor and run inference
	// output := model.Exec([]tf.Output{model.Op("StatefulPartitionedCall", 0)}, map[tf.Output]*tf.Tensor{
	// 	model.Op("serving_default_resnet50_input", 0): tensorInput,
	// })

	// Convert output to classification result
	probabilities := output[0].Value().([][]float32)[0]
	fmt.Println("======================================================")
	fmt.Println(softmaxProb)
	if softmaxProb[0] > softmaxProb[1] {
		c.JSON(http.StatusOK, gin.H{"result": "Not kolam"})
	} else {
		c.JSON(http.StatusOK, gin.H{"result": "Kolam"})
	}
}

func preprocess(img image.Image) [][][][]float32 {
	// Resize image to 224x224
	resized := resize.Resize(224, 224, img, resize.Bilinear)

	// Convert image to tensor and normalize pixel values
	var pixels [][][]float32
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	for y := 0; y < 224; y++ {
		var row [][]float32
		for x := 0; x < 224; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			pixel := []float32{float32(r)/65535, float32(g)/65535, float32(b)/65535}
			for i := range pixel {
				pixel[i] = (pixel[i] - mean[i]) / std[i]
			}
			row = append(row, pixel)
		}
		pixels = append(pixels, row)
	}
	return [][][][]float32{pixels}
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
