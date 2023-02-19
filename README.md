# Image-Classification-Service

# Convert model .h5 to .pb for golang to load using google colab

# Get first layer from model and last layer from model using `saved_model_cli` using this command `saved_model_cli show --dir /path/to/saved_model --all`

# install dependencies
go get github.com/galeone/tfgo
go get github.com/gin-gonic/gin
go get gocv.io/x/gocv
go get github.com/nfnt/resize
go get gonum.org/v1/gonum/floats