# Image-Classification-Service

1. Convert model .h5 to .pb for golang to load using google colab
2. Get first layer from model and last layer from model using `saved_model_cli` using this command `saved_model_cli show --dir /path/to/saved_model --all`
3. install dependencies
    1. go get github.com/galeone/tfgo
    2. go get github.com/galeone/tensorflow/tensorflow/go
    3. go get github.com/gin-gonic/gin
    4. get github.com/nfnt/resize
