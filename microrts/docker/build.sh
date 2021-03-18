VERSION=1.4-cuda10.1-cudnn7-runtime-gym-microrts-0.1.1

docker build  -t cpuheater/gym-microrts:$VERSION  -t cpuheater/gym-microrts:latest -f Dockerfile .

docker push cpuheater/gym-microrts:latest
docker push cpuheater/gym-microrts:$VERSION
