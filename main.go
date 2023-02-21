package main

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/nfnt/resize"

	"example.com/image-classification/docs"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tfgo "github.com/galeone/tfgo"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

// classifyHandler handles image classification.
// @Summary Classify image
// @Description Classifies an uploaded image as kolam or bukan kolam.
// @Accept mpfd
// @Produce json
// @Param image formData file true "Image file to classify"
// @Success 200 {object} map[string]string "Return classification image, kolam / bukan kolam"
// @Failure 400 {object} string "Bad Request"
// @Failure 500 {object} string "Internal Server Error"
// @Router /classify [post]
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

	// Load model
	model := tfgo.LoadModel("saved_model", []string{"serve"}, nil)

	output := model.Exec([]tf.Output{model.Op("StatefulPartitionedCall", 0)}, map[tf.Output]*tf.Tensor{
		model.Op("serving_default_resnet50_input", 0): imageTensor,
	})

	probabilities := output[0].Value().([][]float32)[0]

	softmax_classify := softmax(probabilities)
	if softmax_classify[1] < 0.2694 {
		c.JSON(http.StatusOK, gin.H{"result": "kolam"})
	} else {
		c.JSON(http.StatusOK, gin.H{"result": "bukan kolam"})
	}
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

func main() {
	docs.SwaggerInfo.Title = "Swagger Image Classification API"
	docs.SwaggerInfo.Description = "This is a sample server image classification server."
	docs.SwaggerInfo.Version = "1.0"
	docs.SwaggerInfo.Host = "localhost:8080"
	docs.SwaggerInfo.Schemes = []string{"http", "https"}

	r := gin.Default()

	// Set up classification endpoint
	r.POST("/classify", classifyHandler)
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// Start server
	if err := r.Run(":8080"); err != nil {
		fmt.Println(err)
	}
}
