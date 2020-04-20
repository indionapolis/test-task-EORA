docker build -t eora-test .
docker run -e STREAM_URL=http://192.168.31.172:8080/ -p 80:80 eora-test