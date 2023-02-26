import pandas as pd
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import apis
import requests, json
import numpy as np
import re

tqdm.pandas(desc='(ง •̀_•́)ง')

updated1 = pd.read_csv(r'./收件/06130615收件/06130615月结卡号更新.csv',dtype = str)  #已更新部分1
updated2 = pd.read_csv(r'./收件/06160617收件/06160617月结卡号更新.csv',dtype = str) #已更新部分2
new = pd.read_csv(r'./收件/06180621/06180621收件.csv',sep='\t', dtype = str,error_bad_lines = False)  #新一期原始数据

updated = updated1.append(updated2)

# eff = ['Ture']
# updated = updated[[x in eff for x in ['month_body']]]
updated = updated[updated['month_body'] == 'True'] #更新成功部分

dates = ['20220618','20220619','20220620','20220621']
regions = ['886', '853', '852']
new = new[[x in dates for x in new['b.inc_day']]]
new = new[[x not in regions for x in new['a.city_code']]]
new = new.dropna(subset=['a.customeraccount']) #subset要求列表


new = new.merge(updated,left_on='a.customeraccount', right_on='account',how='left')
rec = new.dropna(subset=['aoiid'])







