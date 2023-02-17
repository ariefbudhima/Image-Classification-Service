package main

import (
	"fmt"
	"image/png"
	"image"
	"net/http"
	"runtime/debug"

	"github.com/gin-gonic/gin"
	"github.com/disintegration/imaging"
	// image "github.com/galeone/tfgo/image"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tfgo "github.com/galeone/tfgo"
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

	// // preprocess the image
	img = imaging.Resize(img, 224, 224, imaging.Lanczos)

	imageTensor, err := imageToTensor(img, 224, 224)

	// Load model
	model := tfgo.LoadModel("saved_model", []string{"serve"}, nil)

	output := model.Exec([]tf.Output{model.Op("StatefulPartitionedCall", 0)}, map[tf.Output]*tf.Tensor{
		model.Op("serving_default_resnet50_input", 0): imageTensor,
	})

	probabilities := output[0].Value().([][]float32)[0]

    // Print the result
    fmt.Printf("Result: %v\n", probabilities)

	fmt.Println(imageTensor)

	c.JSON(http.StatusOK, gin.H{"result":"image sent successfully"})
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


func main() {
	r := gin.Default()

	// Set up classification endpoint
	r.POST("/classify", classifyHandler)

	// Start server
	if err := r.Run(":8080"); err != nil {
		fmt.Println(err)
	}
}
