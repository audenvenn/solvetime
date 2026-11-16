#!/bin/bash
# LM Studio's headless daemon (llmster) as one benchmark server, in the foreground, the way solvetime starts
# every engine: a fresh daemon, ONE model loaded with the given context, LM Studio's OpenAI-compatible server
# on the given port, then its logs until the runner stops this script. The daemon detaches from this process
# group, so stopping it is this script's job: on TERM, INT or exit it unloads, stops the server and the daemon.
#
#   lmstudio.sh <port> <model key> <identifier> <context length>
set -u
LMS="${SOLVETIME_LMS:-$HOME/.lmstudio/bin/lms}"
port="$1"; key="$2"; ident="$3"; ctx="$4"
stop() {
  trap - TERM INT EXIT
  "$LMS" unload --all > /dev/null 2>&1
  "$LMS" server stop > /dev/null 2>&1
  "$LMS" daemon down > /dev/null 2>&1
  kill 0 2> /dev/null        # the log streams
  exit 0
}
trap stop TERM INT EXIT
"$LMS" daemon down > /dev/null 2>&1            # nothing carried over from an earlier run
"$LMS" daemon up || exit 1
"$LMS" runtime ls
"$LMS" log stream --source runtime 2>&1 | sed -u "s/^/[runtime] /" &
"$LMS" log stream --source model --filter output --stats 2>&1 | sed -u "s/^/[model] /" &
echo "### lms load $key --identifier $ident --context-length $ctx --parallel 1"
"$LMS" load "$key" --identifier "$ident" --context-length "$ctx" --parallel 1 -y || exit 1
"$LMS" ps
"$LMS" server start --port "$port" --bind 127.0.0.1 || exit 1
wait
