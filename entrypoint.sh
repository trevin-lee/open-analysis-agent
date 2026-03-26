#!/bin/bash
set -e

# Check for existing valid proxy
PROXY="${X509_USER_PROXY:-/tmp/x509up_u0}"
if [ -f "$PROXY" ]; then
    # Check if proxy is still valid (not expired)
    if voms-proxy-info -file "$PROXY" -exists 2>/dev/null; then
        echo "✓ Valid VOMS proxy found at $PROXY"
        export X509_USER_PROXY="$PROXY"
        # If arguments are passed (e.g., "python -c ..."), run them directly;
# otherwise start the interactive terminal.
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec python -m src.terminal
fi
    fi
    echo "⚠ Proxy at $PROXY has expired"
fi

# If grid certs exist, generate a new proxy
if [ -f /root/.globus/usercert.pem ] && [ -f /root/.globus/userkey.pem ]; then
    echo "Grid certificate found in ~/.globus/"
    echo "Generating VOMS proxy..."
    if voms-proxy-init -voms cms -rfc -valid 192:00; then
        export X509_USER_PROXY="$(voms-proxy-info -path 2>/dev/null || echo /tmp/x509up_u0)"
        echo "✓ VOMS proxy created at $X509_USER_PROXY"
    else
        echo "⚠ voms-proxy-init failed — continuing without grid access"
        echo "  You can retry later: voms-proxy-init -voms cms -rfc"
    fi
else
    echo "No grid certificate found — running without grid access"
    echo "  Mount ~/.globus/ to enable DAS and xrootd"
fi

# If arguments are passed (e.g., "python -c ..."), run them directly;
# otherwise start the interactive terminal.
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec python -m src.terminal
fi
