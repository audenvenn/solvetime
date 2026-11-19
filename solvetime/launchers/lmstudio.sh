#!/bin/bash
# LM Studio's headless daemon (llmster) as one benchmark server, in the foreground, the way solvetime starts
# every engine: a fresh daemon, ONE model loaded with the given context, LM Studio's OpenAI-compatible server
# on the given port, then its logs until the runner stops this script. The daemon detaches from this process
# group, so stopping it is this script's job: on TERM, INT or exit it unloads, stops the server and the daemon.
#
#   lmstudio.sh <port> <model key> <identifier> <context length>
#
# SOLVETIME_LMSTUDIO_GUARDRAILS: its model-loading guardrail for this run. The daemon reads it from
# settings.json (modelLoadingGuardrails.mode) when it starts, so it is written there while the daemon is down,
# and the previous mode is written back once the daemon is down again.
set -u
LMS="${SOLVETIME_LMS:-$HOME/.lmstudio/bin/lms}"
SETTINGS="${SOLVETIME_LMSTUDIO_SETTINGS:-$HOME/.lmstudio/settings.json}"
port="$1"; key="$2"; ident="$3"; ctx="$4"
guard="${SOLVETIME_LMSTUDIO_GUARDRAILS:-}"; previous=""; applied=""
set_guard() {   # mode ("" removes it) -> settings.json; prints the mode it replaces
  python3 - "$SETTINGS" "$1" <<'EOF'
import json, os, sys
path, mode = sys.argv[1], sys.argv[2]
with open(path) as f:
    settings = json.load(f)
guard = settings.setdefault("modelLoadingGuardrails", {})
print(guard.get("mode", ""))
if mode:
    guard["mode"] = mode
else:   # there was none before this run
    guard.pop("mode", None)
with open(path + ".solvetime-tmp", "w") as f:
    json.dump(settings, f, indent=2)
os.replace(path + ".solvetime-tmp", path)
EOF
}
stop() {
  trap - TERM INT EXIT
  "$LMS" unload --all > /dev/null 2>&1
  "$LMS" server stop > /dev/null 2>&1
  "$LMS" daemon down > /dev/null 2>&1
  if [ -n "$applied" ]; then set_guard "$previous" > /dev/null && echo "### guardrails back to ${previous:-unset}"; fi
  kill 0 2> /dev/null        # the log streams
  exit 0
}
trap stop TERM INT EXIT
"$LMS" daemon down > /dev/null 2>&1            # nothing carried over from an earlier run
if [ -n "$guard" ]; then
  previous=$(set_guard "$guard") || exit 1
  applied=1
  echo "### guardrails $guard for this run (was ${previous:-unset})"
fi
"$LMS" daemon up || exit 1
"$LMS" runtime ls
"$LMS" log stream --source runtime 2>&1 | sed -u "s/^/[runtime] /" &
"$LMS" log stream --source model --filter output --stats 2>&1 | sed -u "s/^/[model] /" &
echo "### lms load $key --identifier $ident --context-length $ctx --parallel 1"
"$LMS" load "$key" --identifier "$ident" --context-length "$ctx" --parallel 1 -y || exit 1
"$LMS" ps
"$LMS" server start --port "$port" --bind 127.0.0.1 || exit 1
wait
