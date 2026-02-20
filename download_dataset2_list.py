#!/usr/bin/env python3
"""
Download the provided dataset 2 file list to data/dataset2/.
Uses justice.gov age-verification cookie so DOJ serves the files.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

OUT_DIR = Path("data/dataset2")
DELAY_SEC = 0.8
COOKIE = "justiceGovAgeVerified=true"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Full URLs from your list (PDFs + AVI), one per line
URLS = """
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003208.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003209.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003210.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003211.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003212.avi
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003212.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003213.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003214.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003215.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003216.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003236.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003256.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003276.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003296.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003316.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003317.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003318.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003319.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003321.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003323.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003324.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003325.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003326.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003327.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003328.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003329.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003330.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003331.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003332.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003333.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003334.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003335.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003336.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003337.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003338.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003339.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003340.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003341.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003342.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003343.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003344.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003345.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003346.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003347.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003348.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003349.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003350.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003351.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003352.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003353.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003354.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003355.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003356.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003357.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003358.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003359.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003360.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003361.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003362.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003363.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003364.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003365.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003366.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003367.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003368.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003369.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003370.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003372.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003373.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003374.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003375.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003376.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003377.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003378.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003379.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003381.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003382.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003383.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003384.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003385.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003386.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003387.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003388.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003389.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003390.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003391.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003392.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003393.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003394.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003395.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003396.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003397.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003398.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003399.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003400.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003401.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003402.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003403.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003404.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003405.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003406.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003407.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003408.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003409.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003410.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003411.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003412.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003413.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003414.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003415.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003416.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003417.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003418.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003419.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003420.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003421.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003422.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003423.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003424.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003425.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003426.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003427.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003428.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003429.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003430.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003431.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003432.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003433.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003442.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003443.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003444.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003445.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003446.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003447.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003448.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003449.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003450.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003451.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003452.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003453.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003454.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003455.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003456.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003457.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003458.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003459.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003460.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003461.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003462.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003463.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003464.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003465.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003466.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003467.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003468.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003469.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003470.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003471.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003472.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003473.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003474.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003475.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003476.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003477.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003478.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003479.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003480.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003481.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003482.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003483.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003484.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003485.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003486.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003487.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003488.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003489.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003490.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003491.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003492.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003493.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003494.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003495.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003496.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003497.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003498.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003499.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003500.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003501.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003502.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003503.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003504.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003505.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003506.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003507.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003508.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003509.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003510.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003511.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003512.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003513.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003514.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003515.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003516.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003517.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003518.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003519.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003520.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003521.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003522.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003523.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003524.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003525.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003526.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003527.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003528.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003529.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003530.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003531.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003532.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003533.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003534.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003535.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003536.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003537.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003538.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003539.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003540.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003541.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003542.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003543.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003544.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003545.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003546.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003547.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003548.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003549.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003550.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003551.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003552.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003553.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003554.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003555.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003556.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003557.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003558.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003559.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003560.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003561.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003562.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003563.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003565.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003567.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003569.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003571.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003573.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003575.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003577.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003579.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003581.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003583.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003585.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003587.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003589.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003591.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003593.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003595.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003597.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003599.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003601.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003603.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003604.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003605.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003606.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003607.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003608.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003609.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003610.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003611.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003612.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003613.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003614.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003615.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003616.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003617.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003618.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003619.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003620.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003621.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003622.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003623.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003624.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003625.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003626.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003627.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003628.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003629.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003630.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003631.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003632.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003633.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003634.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003635.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003636.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003637.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003638.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003639.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003640.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003641.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003642.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003643.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003644.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003645.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003646.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003647.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003648.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003649.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003650.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003651.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003652.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003653.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003654.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003655.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003656.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003657.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003658.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003659.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003660.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003661.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003662.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003663.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003664.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003665.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003666.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003667.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003668.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003669.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003670.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003671.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003672.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003673.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003674.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003675.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003676.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003677.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003678.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003679.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003680.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003681.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003682.pdf
https://www.justice.gov/epstein/files/DataSet%202/EFTA00003683.pdf
""".strip().splitlines()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.headers["Cookie"] = COOKIE
    session.headers["Accept"] = "application/pdf,*/*"

    ok = 0
    fail = 0
    for i, url in enumerate(URLS):
        url = url.strip()
        if not url:
            continue
        name = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
        out_path = OUT_DIR / name
        if out_path.exists():
            print(f"[{i+1}/{len(URLS)}] Skip (exists): {name}")
            ok += 1
            continue
        try:
            r = session.get(url, timeout=60, stream=True)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            print(f"[{i+1}/{len(URLS)}] Saved: {name}")
            ok += 1
        except requests.RequestException as e:
            print(f"[{i+1}/{len(URLS)}] Failed {name}: {e}")
            fail += 1
        time.sleep(DELAY_SEC)

    print(f"Done: {ok} saved, {fail} failed -> {OUT_DIR.resolve()}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
