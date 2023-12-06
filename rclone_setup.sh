#!/bin/bash

# ########## Connect remotely to NextCloud (rshare)
if rclone listremotes | grep -q "rshare:" ; then
    echo "Rshare identified as remote. Obscuring password..."

    if rclone about rshare: 2>&1 | grep -q "Used:" ; then
        echo "Successful connection to remote rshare."
    else
        echo "Password needs to be obscured to set up rshare..."
        echo export RCLONE_CONFIG_RSHARE_PASS=$(rclone obscure $RCLONE_CONFIG_RSHARE_PASS) >> /root/.bashrc
        source /root/.bashrc
        if ! rclone about rshare: 2>&1 | grep -q "Used:" ; then
            echo "Error in connecting to remote rshare."; sleep 5
            if [ "$0" != "$BASH_SOURCE" ]; then
                return 1
            else
                exit 1
            fi
        fi
        echo "Connected to remote rshare."
    fi
else
    echo "Rshare not identified as (only) remote. Try to solve manually with AI4EOSC documentation."
    if [ "$0" != "$BASH_SOURCE" ]; then
        return 1
    else
        exit 1
    fi
fi

echo "==================================="
echo "RCLONE SETUP COMPLETE."
echo "==================================="