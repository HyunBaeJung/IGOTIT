@echo off
pushd ¡°%~dp0¡±

scp -i ~/.ssh/ubuntu.pem -r auto ubuntu@54.180.19.36:~

:exit
popd
@echo on