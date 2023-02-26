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
GIS_COOKIE = 'SESSION=c0c5f219-0d3f-4075-ab1a-00e7f15873f2'

data_dispatch = pd.read_csv(r'./06060610派件/06060610派件未识别.csv', sep='\t', dtype=str, error_bad_lines=False)
dates = [str(x) for x in range(20220606, 20220611)]
regions = ['886', '853', '852']
data_dispatch = data_dispatch[[x in dates for x in data_dispatch['a.inc_day']]]
data_dispatch = data_dispatch[[x not in regions for x in data_dispatch['a.req_destcitycode']]]
data_dispatch.dropna(subset=['a.req_addresseeaddr'], inplace=True)
data_dispatch.reset_index(drop=True, inplace=True)

#判断地址是否详细+分词标签
def addr_tag_(splitinfo):
    if pd.isna(splitinfo):
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

tqdm.pandas(desc='地址是否详细打标')
res = data_dispatch.progress_apply(lambda x: addr_tag_(x['a.splitresult']), axis=1) #res储存了return结果：is_detailed, tag, pre

tqdm.pandas(desc='数据结构转换')
res = pd.Series(res).progress_apply(pd.Series) #series转换成df
res.columns = ['is_detailed', 'addr_tag', 'precision']
data_dispatch = data_dispatch.join(res) #将函数返回的结果合并至原始数据
del res #删除res减少内存

data_dispatch = data_dispatch.dropna(subset=['is_detailed']).reset_index(drop=True)  #筛掉无法判断地址详细的数据
detailed = data_dispatch[data_dispatch['is_detailed']].reset_index(drop=True).copy()  #地址详细的数据重命名为detailed
del data_dispatch

#按是否能匹到大组进行分组
grps = detailed.dropna(subset=['a.groupids'])
multi_grps = grps[['$' in x and len(set(x)) > 1 for x in grps['a.groupids']]] #匹到多大组
single_grps = grps[['$' not in x for x in grps['a.groupids']]] #匹到单大组
no_grps = detailed[[pd.isna(x) or len(set(x)) == 1 for x in detailed['a.groupids']]] #匹不到大组

req_grp_freq = []
for grp in tqdm(no_grps.groupby(['a.req_addresseeaddr']), desc='group根据工单地址聚合'): #对无大组的数据，以工单地址为key进行分组
    aoilist = list(grp[1]['b.delivery_xy_aoiid'].dropna())
    sitelist = list(grp[1]['a.finalzc'].dropna())
    city = list(grp[1]['a.city'])[0]
    lgt_avg = grp[1]['b.delivery_lgt'].astype(float).mean()  # 80坐标平均值
    lat_avg = grp[1]['b.delivery_lat'].astype(float).mean()  # 80坐标平均值
    delta = {(abs(x[0] - lgt_avg) + abs(x[1] - lat_avg)): x for x in zip(grp[1]['b.delivery_lgt'].astype(float).values,
                                                                         grp[1]['b.delivery_lat'].astype(float).values)}
    avg_aoi = delta[min(delta.keys())]
    dept = max(set(sitelist), key=sitelist.count) if len(sitelist) > 0 else None ##取频率最高的网点作为组内网点
    if len(aoilist) > 0:  # 筛出组内出现频次最高的80AOI
        pick_aoi = max(set(aoilist), key=aoilist.count)
        top_freq = aoilist.count(pick_aoi)
        aoi_total = len(aoilist)
    else:
        pick_aoi, top_freq, aoi_total = (None, None, None)
    req_grp_freq.append((grp[0], city, dept, grp[1].shape[0], pick_aoi, top_freq, aoi_total, avg_aoi[0], avg_aoi[1]))

tqdm.pandas(desc='数据转换')
req_grp_freq = pd.Series(req_grp_freq).progress_apply(pd.Series)
req_grp_freq.columns = ['req_add', 'grp_city', 'dept', 'grp_freq', '80_aoi_grp', 'top_freq_grp', 'aoi_total_grp', '80_x', '80_y']
# grp_freq即各大组最后确定的80AOI


## 高德api取各组工单地址坐标（跑小量）
# def gd_api(addr: str, city=''):
#     global bar
#     bar.update()
#     url = 'https://restapi.amap.com/v3/place/text?keywords={}&city={}&output=json&offset=20&page=1&key=3bf251b77ec9e398c1d0cb25bdb78977&extensions=all'
#     try:
#         resp = requests.get(url.format(addr, city), timeout=5)
#         resp = json.loads(resp.text)
#         if resp['status'] == '1' and int(resp['count']) > 0:
#             return True, resp['info'], resp['pois'][0]['name'], resp['pois'][0]['location']
#         else:
#             return False, resp['info'], None, None
#     except Exception as e:
#         return False, e, None, None


## 图商api获各组工单地址坐标（跑大量）跑完先把结果另存
def gd_ts(address: str, city: str, retry=3):
    """
    GEO图商获取地址经纬度
    :param retry: 失败重试次数
    :param address: 地址
    :param city: citycode
    :return: 地址对应经纬度，匹配精度，匹配到的地址等级
    """
    if retry == 0:     #停止递归的条件
        return None, None, None
    try:
        url = 'http://gis-ass-mg.sf-express.com/ScriptTool3215302/extension/forward?url=http://gis-int2.int.sfdc.com.cn:1080/geo/api&address={address}&city={city}&opt=gd2&ak=87106f6380af4df0845a693eee58843c'.format(address=address, city=city)
        responce = json.loads(requests.get(url, timeout=5).text)  #jason.loads将网站以dict形式返回                                             # { }代表要传入的参数                                                # format用于打包参数
        if responce["status"] == 0:  #若status==0(搜索成功)，则解析下面几行
            x = responce["result"]['xcoord']
            y = responce["result"]['ycoord']
            pre = responce["result"]['precision']
            level = responce["result"]['match_level']
            return str(x)+','+str(y), pre, level
        else:
            return gd_ts(address, city, retry-1)  #递归(自身调用)，若status != 0,则重新调用该函数，retry-1
    except Exception:
        return gd_ts(address, city, retry-1)

def gd(params): #该函数用于打包参数，便于传入gd_ts
    global bar
    bar.update()
    address, citycode = params
    return gd_ts(address, citycode)

if __name__ == '__main__':
    pool = ThreadPool(16)
    bar = tqdm(total=len(req_grp_freq), desc='图商API获取坐标')
    res = pool.map(gd, zip(req_grp_freq['req_add'].values, req_grp_freq['grp_city'].values)) #将工单地址传入gd_api函数中
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['location', 'precision', 'level'] #location即工单地址的坐标

#正常跑数情况
tmp = res.copy()
req_grp_freq = req_grp_freq.join(tmp)
req_grp_freq = req_grp_freq.dropna(subset=['precision']).reset_index(drop=True)
req_grp_freq.to_csv(r'./06060610工单xy合并.csv',index=False)

# tmp = pd.read_csv(r'./06060610派件/06060610工单xy')
# req_grp_freq = req_grp_freq.join(tmp)
# req_grp_freq = req_grp_freq.dropna(subset=['precision']).reset_index(drop=True)
# req_grp_freq.to_csv(r'./06060610工单xy合并.csv',index=False)

# 筛出精度=2且level>6的部分
req_grp_freq_ = req_grp_freq[req_grp_freq['precision']== 2]
req_grp_freq_ = req_grp_freq_[req_grp_freq_['level'] > 6]
req_grp_freq_ = req_grp_freq_.reset_index(drop=True)

## 坐标落AOI
def get_aoi(location): #传参
    bar.update()
    if location:
        x, y = location.split(',')
        return apis.xy_to_aoi(x, y)
    else:
        return None, None

if __name__ == '__main__':
    pool = ThreadPool(10)
    bar = tqdm(total=len(req_grp_freq_), desc='高德API坐标落AOI')
    res = pool.map(get_aoi, req_grp_freq_['location'].values) #工单地址的坐标
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['gdapi_aoi', 'gdapi_aoiname'] #工单地址的AOI

req_grp_freq_ = req_grp_freq_.join(res) #将结果关联回原数据


# 对比工单AOI和80AOI
## (1) 80AOI==工单地址AOI
grp_update = req_grp_freq_.dropna(subset=['80_aoi_grp']).dropna(subset=['gdapi_aoi']) #筛掉无效数据：80aoi为空，工单aoi为空
grp_update = grp_update[grp_update['80_aoi_grp'] == grp_update['gdapi_aoi']]
gdapi_pick_cnt = len(grp_update) #计算匹配上的个数
print('gdapi与80一致大组个数：\t'+str(gdapi_pick_cnt))

## 筛选剩余待处理数据
grp_not_update = req_grp_freq_.apply(lambda x: pd.isna(x['80_aoi_grp']) or pd.isna(x['gdapi_aoi']) or x['gdapi_aoi'] != x['80_aoi_grp'], axis=1) # 80AOI/工单AOI任一为空,或二者不相等
grp_not_update = req_grp_freq_[list(grp_not_update)]    #将上述掩码打包成list, 放到req_grp_freq中筛出80aoi为空 或 工单aoi为空 或 二者不相等的数据 (实际上等于筛掉了二者相等的数据)
grp_not_update_ = grp_not_update[grp_not_update['80_x'] != 'nan']   #80坐标不空
grp_not_update__ = grp_not_update_.dropna(subset=['gdapi_aoi'])     #80坐标不空 且 工单aoi不空的数据
grp_not_update__ = grp_not_update__[grp_not_update__.gdapi_aoi != 'not_covered']     #80坐标不空 且 工单aoi不空也不为not_covered的数据——>用于30m‘条件(80-xy和工单aoi)

## (2)80附近30m内AOI
def get_aois_around(params):  #定义函数-用于传参
    global bar
    bar.update()
    x, y = params
    return apis.aoi_in_circle(x, y, '30')

if __name__ == '__main__':
    pool = ThreadPool(10)
    bar = tqdm(total=len(grp_not_update__), desc='获取80附近30米AOI列表')
    res = pool.map(get_aois_around, zip(grp_not_update__['80_x'].values, grp_not_update__['80_y'].values))
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['80_aoi_list']  # res = df(x,y,'30')

grp_not_update__.reset_index(drop=True, inplace=True)
grp_not_update__ = grp_not_update__.join(res)
tqdm.pandas(desc='判断gdapi是否落在80附近30米AOI内')
grp_not_update__['gd_in_list'] = grp_not_update__.progress_apply(lambda x: x['gdapi_aoi'] in x['80_aoi_list'] if not pd.isna(x['80_aoi_list']) else False, axis=1) #创建一个新的series(返回一个T/F列),判断gdapi_aoi是否在80_aoi_list内(取交集)

grp_not_update_update = grp_not_update__.query('gd_in_list')   #选list这列中为true的行(即工单aoi和80-30m内aoilist的交集)
#grp_not_update_update = grp_not_update__[grp_not_update__['gd_in_list']]
gd_in_list_cnt = len(grp_not_update_update)  #计算匹配上的个数
print('80附近30米包含gdapi的AOI：\t'+str(gd_in_list_cnt)) #工单AOI和80附近AOI取交集

# #集合可压审补的数据：80aoi==工单aoi, 工单aoi inner 80-30m内
to_update_chkn = grp_update[['req_add','dept','grp_city','gdapi_aoi']].append(grp_not_update_update[['req_add','dept','grp_city','gdapi_aoi']])


## 压审补
def add_chkn(params):
    global bar
    bar.update()
    city, addr, dept, aoiid = params
    url = 'http://gis-ass-aos.sf-express.com:1080/api/addTeam'
    data = {
        "city_code": city,
        "addresses": addr,
        "depts": dept,
        "deptCode": "",
        "type": "2",
        "aoi_id": aoiid,
        "unitId": ""
    }
    header = {
        "Accept": "application/json, text/plain, */*",
        "Origin": "http://gis-ass-aos.sf-express.com:1080",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cookie": GIS_COOKIE
    }
    try:
        resp = requests.post(url=url, data=json.dumps(data), headers=header, timeout=5)
        resp = json.loads(resp.text)
        return resp['message'], resp['status']
    except Exception as e:
        return e, 'Fail'


to_update_chkn.reset_index(drop=True, inplace=True)
if __name__ == '__main__':
    pool = ThreadPool(4)
    bar = tqdm(total=len(to_update_chkn))
    res = pool.map(add_chkn, zip(to_update_chkn['grp_city'].values, to_update_chkn['req_add'].values, to_update_chkn['dept'].values, to_update_chkn['gdapi_aoi'].values))
    res = pd.Series(res).apply(pd.Series)
    res.columns = ['message', 'status']


to_update_chkn = to_update_chkn.join(res)
to_update_chkn_ok = to_update_chkn[to_update_chkn['status']=='ok']
to_update_chkn.to_csv(r'./06060610派件/06060610压审补.csv')





# dept_rslt = []
# for grp in tqdm(no_grps.groupby(['a.req_addresseeaddr']), desc='group根据工单地址聚合'): #对无大组的数据，以工单地址为key进行分组
#     sitelist = list(grp[1]['a.finalzc'].dropna())
#     dept = max(set(sitelist), key=sitelist.count) if len(sitelist) > 0 else None
#     dept_rslt.append((grp[0], dept))
# dept_rslt = pd.Series(dept_rslt).apply(pd.Series)
# dept_rslt.columns = ['req_add', 'dept']
#
# to_update_chkn = to_update_chkn.merge(dept_rslt,on='req_add', how='left')
# to_update_chkn = to_update_chkn.dropna(subset=['dept'])









