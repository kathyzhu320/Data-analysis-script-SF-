
import pandas as pd
from tqdm import tqdm #进度条
from multiprocessing.dummy import Pool as ThreadPool #多线程
import apis  #接口
import requests, json #网站请求
import numpy as np
import re
tqdm.pandas(desc='(ง •̀_•́)ง')


# shift-alt+E 运行
# del ates 删除内存
#【逻辑】：数据清洗————按月结卡号分组+提取组别信息————接口：获装车地址信息(地址、封车坐标)————综合上述信息，按优先级获AOI————接口：取AOI标准地址————关联数据查看卡号AOI贡献率————接口：校验月结库信息防覆盖————接口：更新月结库
CGCS_COOKIE = 'ESSION=dfe221a3-0d8c-46a8-87e2-1500a3c31a50; ua_loginType=cas; rememberUserCookieName=01424467; cgcs-audit1=http://10.244.42.157:8080; sidebarStatus=0'
CMS_COOKIE = 'SESSION=9a9159a0-4f4d-4184-8dbf-4f858e3084d8; ua_loginType=cas; rememberUserCookieName=01424467; sticky=http://10.218.193.99:8080; layoutType=0; theme=1; sidebarStatus=1'

data = pd.read_csv(r'./06100612/06110612收件未识别.csv', sep='\t', dtype=str, error_bad_lines=False)
data = data.dropna(subset=['a.pick_lgt']) #筛掉经度为空的
dates = ['20220610','20220612',] #日期限制
regions = ['886', '853', '852']
bad_data = data[[x not in dates for x in data['a.inc_day']]] #筛出非条件日期
data = data[[x in dates for x in data['a.inc_day']]]  #符合条件的日期
data = data[[x not in regions for x in data['a.citycode']]] #剔除非条件地区港澳台’

tmp = data[['a.waybillno', 'a.pick_lgt','a.pick_lat']]
tmp.columns = ['a.waybillno','x','y']
tmp.to_csv('./06100612/06100612单号xy.csv', index=False) #######另存为临时新文档——到gis oms拿坐标落AOI(id+name)
del tmp

notcall = data[data['a.isnotundercall'] == '1'] #非call
# call = data[data['a.isnotundercall'] != '1']
month_acc = notcall.dropna(subset=['a.customeraccount']) #筛掉notcall中customeraccount为空的
month_acc = month_acc[month_acc['a.customeraccount'] != '1234567890'] #筛掉无效account
month_acc.reset_index(drop=True, inplace=True) #重置索引,修改后不另存为新的df(直接在原有基础上改)
month_acc['a.pick_lgt'] = month_acc['a.pick_lgt'].replace({'0.0': np.nan}) #经纬度为0的用空值替代
month_acc['a.pick_lat'] = month_acc['a.pick_lat'].replace({'0.0': np.nan}) #经纬度为0的用空值替代

res = pd.read_csv(r'./06100612/06100612单号xy_20220614180923372.csv', dtype=str) #导入gis oms跑完AOI后文件
res = res[['a.waybillno', 'aoi_id','aoi_name']] #对比22行
res.columns = ['a.waybillno', 'pickup_aoi','pickup_aoi_name'] #重定义列索引
# res['pickup_aoi'] = res['pickup_aoi'].fillna('not_covered')
month_acc = month_acc.merge(res, on='a.waybillno', how='left') #将能直接得到的AOI及单号并回 至原表(原表中含有无法识别AOI的)

#按月结卡号分组
acc_valid = []
acc_invalid = {}
for g in tqdm(month_acc.groupby('a.customeraccount')):    #按客户类型分组,分组后数据结构为((元组+df),(元组+df),...),g为每一个(元组，df),g[1]即每一个(元组，df)中的df这个元素
    pay_types = list(set(g[1]['b.comb_payment_type_code']))  #将以客户类型为组别中的comb_payment信息转换成set集合形式以去重（集合内无重复元素）
    zones = list(set(g[1]['a.zonecode']))  #
    pick_aois = list(g[1]['pickup_aoi'])
    pick_aoi = max(set(pick_aois), key=pick_aois.count)
    waybill_nos = list(g[1]['a.waybillno'][0:5])
    citycode = list(g[1]['a.citycode'])[0]
    if len(pay_types) == 1 and not pd.isna(pay_types[0]) and ('寄付' in pay_types[0] and '第三方' not in pay_types[0]) and len(zones)==1: #若paytypes有且只有一个元素，且该元素不为NaN，且该元素为寄付而非第三方，且网点唯一
        acc_valid.append((citycode, g[0], pick_aoi, waybill_nos, zones[0])) # g[0]=卡号，将上述处理过的数据合并至acc_valid
    else:
        acc_invalid[g[0]] = (pay_types, len(zones))

acc_used = pd.Series(acc_valid).apply(pd.Series) #固定用法：将序列转换成df，apply将元组拆分并回series
acc_used.columns = ['citycode', 'account', 'pick_aoi', 'waybill_no', 'dept_code']

#36装车地址信息
def fvp(no):  #在fvp系统获取运单封车地址（文档调用import apis）
    res = apis._get_fvp_info(no)
    ad = None
    coord = None
    cus = None
    first1 = True
    first2 = True
    if type(res) is list:
        for item in res:
            if item['opCode'] == '36' and first1:
                first1 = False
                if 'exts' in item:
                    if 'ext25' in item['exts']:
                        ad = item['exts']['ext25']
                    if 'ext24' in item['exts']:
                        coord = item['exts']['ext24']
            if item['opCode'] == '302' and first2:
                first2 = False
                if 'exts' in item:
                    if 'ext13' in item['exts']:
                        cus = item['exts']['ext13']
    return ad, coord, cus   #地址、封车坐标、客户信息


def getfvp(waybills): #更新多线程进度条
    global bar
    bar.update()
    for w in waybills:
        if not pd.isna(fvp(w)[0]):  #调用上面的fvp()，fvp(w)[0]即return结果中的ad,若ad不为零
            return fvp(w)[0], fvp(w)[1] #仅取fvp()return中的ad,coord
    return None, None


if __name__ == '__main__': #多线程
    pool = ThreadPool(8) #线程池(数量)
    bar = tqdm(total=len(acc_used)) #进度条
    res = pool.map(getfvp, acc_used['waybill_no'].values) #调用getfvp()
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['ad', 'coord']  #得到ad,coord,并更新至res

acc_used = acc_used.join(res) #将res横向合并至acc_used (join需确保数据来源一致，否则索引会出问题)

sites = pd.read_csv(r'./sites.csv', sep='\t', dtype=str)
sites = sites[['网点代码', '网点地址', '网点经度', '网点纬度']]
acc_used = acc_used.merge(sites, left_on='dept_code', right_on='网点代码', how='left')

#按优先级获取AOI
def get_aoi(params):    # params传参(以元组形式打包多个参数传入函数)
    city, pick_aoi, coord, zaddr, zlng, zlat = params #打包成元组
    global bar
    bar.update()
    aoi = None
    tag = None
    try:
        if not pd.isna(pick_aoi) and pick_aoi != 'not_covered': #54收件aoi
            aoi = pick_aoi
            tag = 'pickup' # AOI来源标签
            return aoi, tag
        if not pd.isna(coord):  #封车地址
            aoi = apis.xy_to_aoi(coord.split('|')[0], coord.split('|')[1], 3)[0]
            tag = 'fvp'
        if (pd.isna(aoi) or aoi == 'not_covered') and not pd.isna(zlng):#网点坐标
            aoi = apis.xy_to_aoi(zlng, zlat, 3)[0]
            tag = 'site_coord'
        if pd.isna(aoi) or aoi == 'not_covered':#网点地址
            tag = None
            at = apis.atpai(zaddr, city, 2) #atpai直接地址拿AOI
            if not pd.isna(at) and 'count' in at and at['count'] > 0 and 'aoiid' in at['tcs'][0]:
                aoi = at['tcs'][0]['aoiid']
                tag = 'site_addr'
        return aoi, tag
    except Exception:
        return None, None


if __name__ == '__main__': #设置多线程跑get_aoi()
    pool = ThreadPool(16)
    bar = tqdm(total=len(acc_used))
    res = pool.map(get_aoi, zip(acc_used['citycode'].values, acc_used['pick_aoi'].values, acc_used['coord'].values, acc_used['网点地址'].values, acc_used['网点经度'].values, acc_used['网点纬度'].values))
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['aoiid', 'tag'] #取得aoiid,tag(aoi来源)

acc_used = acc_used.join(res)
acc_used_no_aoi = acc_used[[pd.isna(x) or x == 'not_covered' for x in acc_used['aoiid']]]
acc_used = acc_used.query('aoiid != \'not_covered\'').dropna(subset=['aoiid'])
acc_used.reset_index(drop=True, inplace=True)

#取AOI下标准地址
def get_aoi_addr(params): #调用接口,文档
    citycode, zonecode, aoiid, page, retry = params
    global bar
    if retry == 0 or page == 4:
        bar.update()
        return None
    header = {
        'Accept': 'application/json, text/plain, */*',
        'X-Requested-With': 'XMLHttpRequest',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
        'Referer': 'http://gisasscmsbg-gis-ass-cms.dcn2.k8s.sf-express.com/cms/schAoi/schAoiEdit',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cookie': CMS_COOKIE
    }
    url = 'http://gisasscmsbg-gis-ass-cms.dcn2.k8s.sf-express.com/cms/address/queryAddressByAoi?cityCode={}&znoCode={}&aoiId={}&pageSize=100&pageNo={}&address=&total=354'
    try:
        resp = requests.get(url.format(citycode, zonecode, aoiid, page), headers=header, timeout=10)
        resp = json.loads(resp.text)
        if resp['code'] == 200 and 'data' in resp and 'records' in resp['data']:
            for add in resp['data']['records']:
                if add['type'] == 1: # 取AOI下第一条标准地址
                    bar.update()
                    return add['address']
            return get_aoi_addr((citycode, zonecode, aoiid, page + 1, retry))
        else:
            return get_aoi_addr((citycode, zonecode, aoiid, page, retry - 1))
    except Exception:
        return get_aoi_addr((citycode, zonecode, aoiid, page, retry - 1))


if __name__ == '__main__':
    pool = ThreadPool(8)
    bar = tqdm(total=len(acc_used))
    res = pool.map(get_aoi_addr, zip(acc_used['citycode'].values, acc_used['dept_code'].values, acc_used['aoiid'].values, ['1'] * len(acc_used), [3] * len(acc_used)))
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['addr']

acc_used = acc_used.join(res)
to_update = acc_used.dropna(subset=['addr'])
to_update.to_csv(r'./06100612/待更新卡号.csv', index=False) #可更新AOI的卡号集

# #关联数据查看卡号AOI贡献率
# to_update = pd.read_csv(r'./06100612/待更新卡号.csv', dtype=str)
# test = pd.read_csv(r'./06080610/month_testt.csv', dtype=str)
# new = pd.read_csv(r'./06080610/06080610收件.csv', sep='\t', dtype=str, error_bad_lines=False)  #新日期的数据
# dates = ['20220608','20220610']
# regions = ['886', '853', '852']
# new = new[[x in dates for x in new['a.inc_day']]]
# new = new[[x not in regions for x in new['a.citycode']]]
# new = new.merge(test, left_on='a.customeraccount', right_on='account', how='left')
# new_rec = new.dropna(subset=['account'])  #新识别
# new_rec_c = new_rec[new_rec['a.citycode'] == new_rec['citycode']]
# new_rec_call = new_rec[new_rec['a.isnotundercall'] != '1']

#校验:查询月结库信息,以防覆盖仓管数据
def get_monthacc(params):  #调用接口-文档
    area, city, account = params
    global bar
    bar.update()
    header = {
        'Accept': 'application/json, text/plain, */*',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
        'Referer': 'https://gis-aos-cgcs.sf-express.com/audit/monthlyStatistic/monthlyBalanceFeedBack',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cookie': CGCS_COOKIE  #cms-区域-地址管理（城市+地址名称）-ctrl+shift+I
    }
    url = 'https://gis-aos-cgcs.sf-express.com/audit/monthlyAccount/searchAccount?id=&areaCode={}&cityCode={}&znoCode=&source=2&pageNo=1&pageSize=10&sendAddress=&account={}'
    try:
        resp = requests.get(url=url.format(area, city, account), headers=header, timeout=5)
        resp = json.loads(resp.text)
        if resp['code'] == 200 and resp['data']['total'] > 0:
            rslt = resp['data']['rows'][0]
            return rslt['znoCode'], rslt['tag'], rslt['updateBy'], rslt['delFlag'],rslt['atAoiId']
        else:
            return None, None, None, None, None
    except Exception:
        return None, None, None, None, None


cities = pd.read_csv(r'./citycode.csv', dtype=str)
areas = cities[['city_code', 'area_code']]
areas.columns = ['citycode', 'area_code']
to_update = to_update.merge(areas, on='citycode', how='left')


if __name__ == '__main__': 
    pool = ThreadPool(8)
    bar = tqdm(total=len(to_update))
    res = pool.map(get_monthacc, zip(to_update['area_code'].values, to_update['citycode'].values, to_update['account'].values))
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['znoCode','mon_tag','updateBy','delFlag','atAoiId']

to_update_ = to_update.copy()
to_update_ = to_update_.join(res)
to_update_ = to_update_.dropna(subset=['znoCode'])
to_update_ = to_update_[[x =='' for x in to_update_['atAoiId']]]


#更新月结库：卡号+地址
def updateAccount(params): #调用接口-文档
    acc, zno, new_ad = params
    global bar
    bar.update()
    url = 'http://gis-aos-cgcs.sit.sf-express.com/audit/api/monthlyAccount/updateAccount'
    test = {
        'account': acc,
        'znoCode': zno,
        'address': '',
        'newAddress': new_ad,
        'contact': '',
        'phone': '',
        'addressType': ''
    }
    header = {'Content-Type': 'application/json'}
    try:
        resp = json.loads(requests.post(url, data=json.dumps(test), headers=header).text)
        if resp['code'] == 200:
            return True
        else:
            return 'fail'
    except Exception as e:
        return e

if __name__ == '__main__':
    pool = ThreadPool(4)
    bar = tqdm(total=len(to_update_))
    res = pool.map(updateAccount, zip(to_update_['account'].values, to_update_['dept_code'].values, to_update_['addr'].values))
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['month_body']

to_update_.reset_index(drop=True, inplace=True)
to_update_ = to_update_.join(res)

#备份
to_update_.to_csv(r'./06100612/月结卡号更新.csv', index=False)





