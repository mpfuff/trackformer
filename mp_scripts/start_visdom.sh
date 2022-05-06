export VISDOM_USERNAME=mobe
export VISDOM_PASSWORD=interpc
export VISDOM_COOKIE=~/.visdom/cookie
VISDOM_USE_ENV_CREDENTIALS=1 visdom -enable_login -force_new_cookie -port 8090
