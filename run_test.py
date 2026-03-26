"""Test run: Z→μμ toy analysis end-to-end."""

import os
from pathlib import Path

# Load .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from openanalysisagent.agent import create_cms_agent

agent = create_cms_agent()

prompt = """\
Generate toy dimuon data mimicking a Z→μμ signal (peaked at 91.2 GeV, width ~2.5 GeV, \
500 signal events) on a flat background (~50 events per bin across 60-120 GeV). Then:

1. Fill a histogram of the invariant mass from 60 to 120 GeV with 30 bins
2. Build a pyhf model with the Z peak as signal and the flat background, \
   with 10% background normalization uncertainty
3. Run a maximum likelihood fit
4. Compute the discovery significance
5. Set a 95% CLs upper limit on the signal strength
6. Make a CMS-style plot of the mass distribution labeled "Z → μμ" \
   with Private Work label, save to /tmp/zmumu_mass.png
7. Make a Brazil band plot from the limit scan, save to /tmp/brazil.png

Show me all the numerical results.
"""

result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

# Print the final agent response
for msg in result["messages"]:
    if hasattr(msg, "content") and msg.content and msg.type == "ai":
        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
            print(msg.content)
