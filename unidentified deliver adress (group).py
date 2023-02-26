import pandas as pd
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import apis
import requests, json
import numpy as np
import re

tqdm.pandas(desc='(ง •̀_•́)ง')



CMS_COOKIE = 'SESSION=6d5dc0e2-1113-4722-a563-c5b3a3b1f5e0; ua_loginType=cas; rememberUserCookieName=01424467; sticky=http://10.218.193.99:8080; layoutType=0; theme=1; sidebarStatus=1'
CGCS_COOKIE = 'SESSION=bed56ffa-2901-40c7-8a18-2dddbef13dd4; ua_loginType=cas; rememberUserCookieName=01424467; cgcs-audit1=http://10.244.42.157:8080; sidebarStatus=0'

data_dispatch = pd.read_csv(r'./06110613派件/061106013派件未识别.csv', sep='\t', dtype=str, error_bad_lines=False)
dates = [str(x) for x in range(20220611, 20220614)]
regions = ['886', '853', '852']
data_dispatch = data_dispatch[[x in dates for x in data_dispatch['a.inc_day']]]
data_dispatch = data_dispatch[[x not in regions for x in data_dispatch['a.req_destcitycode']]]
data_dispatch.dropna(subset=['a.req_addresseeaddr'], inplace=True)  #筛掉无工单地址的
data_dispatch.reset_index(drop=True, inplace=True)

#判断地址是否详细+分词标签
def addr_tag_(splitinfo):
    if pd.isna(splitinfo): #若分词信息为空，则返回None
        return None, None, None
    mask = [False] * 20  #28-31初始值设置
    lvls = []
    pre = 36
    tag = ''
    for item in splitinfo.split(';')[0].split('|'):
        lvl = item.split('^')[1][1:]
        if int(lvl) < 19:
            mask[int(lvl)] = True
        if lvl not in lvls:
            lvls.append(lvl)
            if int(lvl) < 11:
                pre -= int(lvl)
    is_detailed = (mask[9] and mask[11]) or (mask[13] and (mask[5] or mask[9] or mask[10]))  #注意and/or为判断语句，判断地址是否详细，因此在is_detailed该列返回的是bool
    if mask[10] and mask[13]:
        tag += '(10+13)'
    elif mask[9] and mask[13]:
        tag += '(9+13)'
    elif mask[9] and mask[11]:
        tag += '(9+11)'
    elif mask[5] and mask[13]:
        tag += '(5+13)'
    elif mask[3] and mask[13]:
        tag += '(3+13)'
    elif mask[13]:
        tag += '(single_13)'
    return is_detailed, tag, pre


tqdm.pandas(desc='地址是否详细打标')  #进度条(desc'名称')
res = data_dispatch.progress_apply(lambda x: addr_tag_(x['a.splitresult']), axis=1)    #apply即 将后面的内容(无名函数lambda x)应用到data_dispatch，lambda又调用了函数addr_tag_# apply前的progress用于显示进度条，与.apply并无差
# 等价于 res = data_dispatch['a.splitresult'].progress_apply(addr_tag_)
# 理解：对data_dispatch['a.splitresult']的每一行调用一次函数
# res中储存的是对data_dispatch['a.splitresult']进行addr_tag_函数处理后的return结果：is_detailed, tag, pre
'''
lambda x 相当于声明了一个无名函数tmp_func(x)
即等效于： 
def tmp_func(x):
    addr_tag_(x['a.splitresult']) #62-63的x = apply前面的对象data_dispatch，x['a.splitresult'] = data_dispatch['a.splitresult']
    
res = []
for line in data_dispatch:
    res.append(tmp_func(line))
'''
tqdm.pandas(desc='数据格式转换') #数据结构转换
res = pd.Series(res).progress_apply(pd.Series) #series转换成df
res.columns = ['is_detailed', 'addr_tag', 'precision']
data_dispatch = data_dispatch.join(res) #将函数返回的结果合并至原始数据
del res #删除res减少内存

data_dispatch = data_dispatch.dropna(subset=['is_detailed']).reset_index(drop=True)
detailed = data_dispatch[data_dispatch['is_detailed']].reset_index(drop=True).copy()  #地址详细的数据
del data_dispatch

grps = detailed.dropna(subset=['a.groupids']) #删掉无大组ID的数据
multi_grps = grps[['$' in x and len(set(x)) > 1 for x in grps['a.groupids']]] #拎出匹配到多大组的数据
single_grps = grps[['$' not in x for x in grps['a.groupids']]] #拎出匹配到单大组的数据
no_grps = detailed[[pd.isna(x) or len(set(x)) == 1 for x in detailed['a.groupids']]] #################未匹配到大组的数据

grp_freq = [] ###############大组聚合
for grp in tqdm(single_grps.groupby(['a.groupids']), desc='group聚合处理'): #对匹配到单大组的数据，以groupsid为key进行分组
    aoilist = list(grp[1]['b.delivery_xy_aoiid'].dropna()) #筛选同时包有80aoi和80坐标的（以80aoi为字符进行drop,无aoi一定无坐标)，剩余数据打包成list
    city = list(grp[1]['a.city'])[0]
    lgt_avg = grp[1]['b.delivery_lgt'].astype(float).mean()  #80坐标平均值
    lat_avg = grp[1]['b.delivery_lat'].astype(float).mean()  #80坐标平均值
    delta = {(abs(x[0] - lgt_avg) + abs(x[1] - lat_avg)): x for x in zip(grp[1]['b.delivery_lgt'].astype(float).values,
                                                                         grp[1]['b.delivery_lat'].astype(float).values)} #给组内坐标算平均值偏差
    avg_aoi = delta[min(delta.keys())]  #在各groupids组内所有的80坐标里，选择一个离该组80坐标平均值最接近的，作为该组的80坐标
    if len(aoilist) > 0:  #筛出组内出现频次最高的80AOI
        pick_aoi = max(set(aoilist), key=aoilist.count)
        top_freq = aoilist.count(pick_aoi)
        aoi_total = len(aoilist)
    else:
        pick_aoi, top_freq, aoi_total = (None, None, None)
    grp_freq.append((grp[0], city, grp[1].shape[0], pick_aoi, top_freq, aoi_total, avg_aoi[0], avg_aoi[1]))

tqdm.pandas(desc='数据转换')
grp_freq = pd.Series(grp_freq).progress_apply(pd.Series)
grp_freq.columns = ['group', 'grp_city', 'grp_freq', '80_aoi_grp', 'top_freq_grp', 'aoi_total_grp', '80_x', '80_y']
# grp_freq即各大组最后确定的80AOI

def get_group(groupid: str, citycode: str, retry: int): #取大组标准地址+aoi+kw
    if retry == 0:
        return None
    try:
        url = 'http://gis-cms-bg.sf-express.com/cms/api/address/getAddrByCityCodeAndAddr?ticket=' \
              'ST-1609819-CtBGlY0DaSTKiWKkFSDf-casnode1&cityCode={}&addressId={}'.format(citycode, groupid)
        responce = json.loads(requests.get(url, timeout=10).text)
        if responce['code'] == 200:
            return responce['data']
        else:
            return get_group(groupid, citycode, retry-1)
    except Exception:
        return get_group(groupid, citycode, retry-1)


def grp(params): #传参函数：便于调用多线程的pool.map(func,zip())
    grp, city = params
    global bar
    bar.update()
    resp = get_group(grp, city, 5)
    std, aoi, kw = None, None, None
    if resp:
        std = resp['address']
        aoi = resp['aoiId']
        kw = resp['keyword']
    return std, aoi, kw


if __name__ == '__main__':
    pool = ThreadPool(8)
    bar = tqdm(total=len(grp_freq), desc='获取大组标准地址及AOI')
    res = pool.map(grp, zip(grp_freq['group'].values, grp_freq['grp_city'].values))
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['std_addr', 'grp_aoi', 'grp_kw']  #大组标准地址、大组标准地址的AOI、大组名关键词

grp_freq = grp_freq.join(res)
tmp = grp_freq[[pd.isna(x) for x in grp_freq['std_addr']]].reset_index(drop=True)
del res

if __name__ == '__main__': #防止漏跑
    pool = ThreadPool(8)
    bar = tqdm(total=len(tmp))
    res = pool.map(grp, zip(tmp['group'].values, tmp['grp_city'].values))
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['std_addr', 'grp_aoi', 'grp_kw']

tmp['std_addr'] = res['std_addr']
tmp['grp_aoi'] = res['grp_aoi']
grp_freq = grp_freq.dropna(subset=['std_addr']).append(tmp.dropna(subset=['std_addr']), sort=False).reset_index(drop=True)
grp_freq = grp_freq[grp_freq['grp_aoi'] == ''].reset_index(drop=True) #聚合大组AOI为空的


def gd_api(addr: str, city=''): #高德api取坐标
    global bar
    bar.update()
    url = 'https://restapi.amap.com/v3/place/text?keywords={}&city={}&output=json&offset=20&page=1&key=3bf251b77ec9e398c1d0cb25bdb78977&extensions=all'
    try:
        resp = requests.get(url.format(addr, city), timeout=5)
        resp = json.loads(resp.text)
        if resp['status'] == '1' and int(resp['count']) > 0:
            return True, resp['info'], resp['pois'][0]['name'], resp['pois'][0]['location']
        else:
            return False, resp['info'], None, None
    except Exception as e:
        return False, e, None, None


def get_aoi(location): #坐标落AOI
    bar.update()
    if location:
        x, y = location.split(',')
        return apis.xy_to_aoi(x, y)
    else:
        return None, None


if __name__ == '__main__':
    pool = ThreadPool(10)
    bar = tqdm(total=len(grp_freq), desc='高德API获取坐标')
    res = pool.map(gd_api, grp_freq['std_addr'].values) #大组标准地址
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['success', 'info', 'gd_poi', 'location'] #大组标准地址的坐标

tmp = res.copy()

if __name__ == '__main__':
    pool = ThreadPool(10)
    bar = tqdm(total=len(tmp), desc='高德API坐标落AOI')
    res = pool.map(get_aoi, tmp['location'].values) #大组标准地址的坐标
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['gdapi_aoi', 'gdapi_aoiname'] #大组标准地址的AOI

tmp = tmp.join(res)
grp_freq = grp_freq.join(tmp)

#对比工单AOI和80AOI
grp_update = grp_freq.dropna(subset=['80_aoi_grp']).dropna(subset=['gdapi_aoi'])#筛掉无效数据

##(1) 80aoi == 工单地址aoi
grp_update = grp_update[grp_update['80_aoi_grp'] == grp_update['gdapi_aoi' ]] #80AOI=标准地址AOI
#grp_update_['校验网点'] = 0
gdapi_pick_cnt = len(grp_update)   #字符串计算匹配个数
print('gdapi与80一致大组个数：\t'+str(gdapi_pick_cnt))

#任意为空/两者不相等
grp_not_update = grp_freq.apply(lambda x: pd.isna(x['80_aoi_grp']) or pd.isna(x['gdapi_aoi']) or x['gdapi_aoi'] != x['80_aoi_grp'], axis=1) # 80AOI/工单AOI任一为空,或二者不相等，返回的是T/F的掩码
grp_not_update = grp_freq[list(grp_not_update)]     #将上述掩码打包成list, 放到grp_freq中筛出80aoi为空 或 标准地址aoi为空 或 二者不相等的数据 (实际上等于筛掉了二者相等的数据)
grp_not_update_ = grp_not_update[grp_not_update['80_x'] != 'nan']   #80坐标不空
grp_not_update__ = grp_not_update_.dropna(subset=['gdapi_aoi'])     #80坐标不空 且 标准aoi不空的数据
grp_not_update__ = grp_not_update__[grp_not_update__.gdapi_aoi != 'not_covered']      #80坐标不空 且 工单aoi不空 且 工单aoi不为not_covered的数据——>用于30m‘条件


def get_aois_around(params): #80附近30m内AOI
    global bar
    bar.update()
    x, y = params
    return apis.aoi_in_circle(x, y, '30')

if __name__ == '__main__':
    pool = ThreadPool(10)
    bar = tqdm(total=len(grp_not_update__), desc='获取80附近30米AOI列表')
    res = pool.map(get_aois_around, zip(grp_not_update__['80_x'].values, grp_not_update__['80_y'].values))
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['80_aoi_list'] # res = df(x,y,'30')


grp_not_update__.reset_index(drop=True, inplace=True)
grp_not_update__ = grp_not_update__.join(res) #80list关联原数据
tqdm.pandas(desc='判断gdapi是否落在80附近30米AOI内')
grp_not_update__['gd_in_list'] = grp_not_update__.progress_apply(lambda x: x['gdapi_aoi'] in x['80_aoi_list'] if not pd.isna(x['80_aoi_list']) else False, axis=1) #创建一个新的series(返回一个T/F列),判断gdapi_aoi是否在80_aoi_list内(取交集)


grp_not_update_update = grp_not_update__.query('gd_in_list') #选list这列中为true的行
#grp_update_['校验网点'] = 1
gd_in_list_cnt = len(grp_not_update_update)  #计算匹配上的个数
print('80附近30米包含gdapi的AOI：\t'+str(gd_in_list_cnt)) #工单AOI和80附近AOI取交集

grp_update_ = grp_update[['group', 'grp_city', 'gdapi_aoi']].append(grp_not_update_update[['group', 'grp_city', 'gdapi_aoi']])
#grp_update_ = grp_update[['group', 'grp_city', 'gdapi_aoi','校验网点']].append(grp_not_update_update[['group', 'grp_city', 'gdapi_aoi','校验网点']])
grp_update_.columns = ['地址ID', '城市编码', 'AOIID']
grp_update_['校验网点'] = 1
grp_update_['AOI单元ID'] = None
grp_update_.to_csv(r'./06110613派件/06110613派件未识别更新大组.csv', index=False)
print('待更新大组数据已保存...')

not_gd = grp_not_update.append(grp_not_update_update, sort=False).drop_duplicates(subset=['group'], keep=False)
not_gd.to_csv(r'./06110613派件/06110613gdapi不能处理数据.csv', index=False)









#####**测试更新成功的大组数量
grp_update_= pd.read_csv(r'./06180621待更新大组.csv', dtype=str, error_bad_lines=False)

def test(addrid,citycode): #CMS-地址管理查询地址id
    # global bar
    # bar.update()
    header = {
        "Accept": "application/json, text/plain, */*",
        "Origin": "http://gis-ass-aos.sf-express.com:1080",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep - alive",
        "Cookie": "SESSION=1459d9b9-f563-40c0-9ae3-9330114866b8; ua_loginType=cas; rememberUserCookieName=01424467; sticky=http://10.218.192.93:8080; layoutType=0; theme=1"
    }
    url = 'http://gisasscmsbg-gis-ass-cms.dcn2.k8s.sf-express.com/cms/address/list?aoiIds=&addressId={}&znoCode=&aoiId=&unitId=&unitCode=&aoiSource=&cityCode={}&address=&status=&type=&tags=&adcode=&pageSize=10&pageNo=1&noZoneCodeCheck=false&noSchCodeCheck=false&noAoiCheck=false&noAoiUnitCheck=false&noTagCheck=false&noLockCheck=false&freqCheck=false&znoDiffCheck=false&invalidZnoCheck=false&schCode=&noZnoCode=0&noSch=0&noAoi=0&noAoiUnit=0&isFreq=0&isZnoDiff=0&invalidZno=0&noTag=0&noLock=0'
    try:
        res = requests.get(url.format(addrid,citycode),headers=header,timeout=5)
        res = json.loads(res.text)
        if res['code'] == 200 and res['data']['rows'] != '':
            for item in res['data']['rows']:
                if item['aoiId'] != '':
                    return item['aoiId']
                else:
                    return None
        else:
            return  None
    except Exception:
        return None

result = grp_update_.progress_apply(lambda x: test(x['地址ID'],x['城市编码']),axis=1)
result.columns = ['result']
grp_update_['result'] = result

updated = grp_update_.dropna(subset=['result']) #更新成功的数量

updated.to_csv(r'./更新成功大组.csv')