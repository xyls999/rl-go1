#!/usr/bin/env bash
echo '--- lock files'
ls -la "$HOME/.nvidia-omniverse" 2>/dev/null | head -8
find "$HOME/.nvidia-omniverse" "$HOME/.cache/ov" -name '*.lock' -o -name 'lock' 2>/dev/null | head -10
echo '--- recent files in cache'
ls -lat "$HOME/.cache/ov" 2>/dev/null | head -6
echo '--- kit data'
ls -lat /root/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/data 2>/dev/null | head -5
echo '--- tmp omni'
ls -lat /tmp | grep -aiE 'omni|kit|carb' | head -5