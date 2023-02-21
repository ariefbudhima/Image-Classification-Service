# Image-Classification-Service

<<<<<<< HEAD
# Convert model .h5 to .pb for golang to load using google colab

# Get first layer from model and last layer from model using `saved_model_cli` using this command `saved_model_cli show --dir /path/to/saved_model --all`

# install dependencies
go get github.com/galeone/tfgo
go get github.com/gin-gonic/gin
go get gocv.io/x/gocv
go get github.com/nfnt/resize
go get gonum.org/v1/gonum/floats
=======
1. Convert model .h5 to .pb for golang to load using google colab
2. Get first layer from model and last layer from model using `saved_model_cli` using this command `saved_model_cli show --dir /path/to/saved_model --all`
3. install dependencies
    1. go get github.com/galeone/tfgo
    2. go get github.com/galeone/tensorflow/tensorflow/go
    3. go get github.com/gin-gonic/gin
    4. get github.com/nfnt/resize
>>>>>>> 7c3d2ea10a54fa70bbc599fc9ca35cb7c4206cc1
