#!/bin/sh
usage() {
  echo "実行するには2個の引数が必要です。
  第一引数: 監視対象ファイル名
  第二引数: 監視対象ファイルが更新された際に実行されるコマンド
  例： ./autoexec.sh a.cpp 'g++ a.cpp && ./a.cpp'"
}
update() {
  echo `openssl sha256 -r $1 | awk '{print $1}'`
}
if [ $# -ne 2 ];
then
  usage
  exit 1
fi

lockfile=watch.lock

mkdir ${lockfile} > /dev/null 2>&1
if [ $? -ne 0 ];then
  echo "Process is already runnning"
  exit 1
fi

trap 'rmdir ${lockfile} 2>&1; exit' INT TERM

INTERVAL=1 #監視間隔, 秒で指定
last=`update $1`
echo "Watching... file: $1"
while true;
do
  sleep $INTERVAL
  current=`update $1`
  lsof=`lsof -f -- $1 2>/dev/null`
  if [ "$last" != "$current" ]; then
    nowdate=`date '+%Y/%m/%d'`
    nowtime=`date '+%H:%M:%S'`
    echo "$nowdate $nowtime : detected update of file: $1"
    if [ -n "$lsof" ]; then
      echo "=> Skipping, $1 is opened by other process..."
    else
      echo "=> Executing Command : $2"
      eval $2
      echo "\nWatching... file: $1"
    fi
    last=$current
  fi
done
