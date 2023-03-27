
sudo -u smdi $PWD/updateSMDI.csh py38.1


################################################################################
#creating a docker container on windows:
docker pull continuumio/anaconda3
docker image ls
type .\requirements_dev.yaml|Select-String  -Pattern "- -e ." -NotMatch -SimpleMatch >.\requirements_docker.yaml
docker build -t registry.code.roche.com/smdd/python/ml_qm_n:latest .
docker run -it registry.code.roche.com/smdd/python/ml_qm_n

# gitlab tocken with all permissions: Js9Pcg9niWUgpjuDV6sy
docker login registry.code.roche.com
usename: albertgo
password: key
docker push registry.code.roche.com/smdd/python/ml_qm_n
