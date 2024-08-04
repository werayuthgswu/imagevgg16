cp ../tech44all_token.json tensorflow/token.json
tar -cvf tensorflow.tar tensorflow
docker build -t vgg16-api:latest .
