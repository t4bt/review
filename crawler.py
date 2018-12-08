import sys
from fake_useragent import UserAgent
import requests
import lxml.html
import mojimoji
import time
import mysql.connector
import random
import pickle
import codecs
import math
import datetime
import os
import pandas as pd
codecs.register(lambda name: codecs.lookup('utf8') if name == 'utf8mb4' else None)


class CrawlerCore:

    def __init__(self):
        self.ua = UserAgent()
        self.ua_list = [self.ua.chrome,
                        self.ua.firefox,
                        self.ua.edge]

    #URLからHTMLを取得する
    def get_html(self, url):
        header = {'User-Agent':str(self.ua_list[random.randint(0,2)])}
        try:
            res = requests.get(url, headers=header)
            html = res.content
            code = res.status_code
            if code == 503:
                print(code)
        except:
            html = '<html></html>'
        return html

    #HTMLからXPATHで指定した個所を取得
    @staticmethod
    def get_xpath(html, xpath):
        return lxml.html.fromstring(html).xpath(xpath)

class Crawler_AmazonReview(CrawlerCore):

    def __init__(self):
        super().__init__()
        self.CC = CrawlerCore()

    def get_item_url_list(self, url, n_page):
        tlist = [1,3,5,7,9,11]
        item_url_list = []
        for i in range(1,n_page+1):
            html = self.CC.get_html(url.format(int(i)))
            item_list = self.CC.get_xpath(html, '//*/ul//li/div//div/div[1]/a[@class="a-link-normal s-access-detail-page  s-color-twister-title-link a-text-normal"]/@href')
            if len(item_list) == 0:
                print("{}:Error".format(i))
            else:
                print("{}:Completed {}".format(i, len(item_list)))
            item_url_list += item_list
            time.sleep(tlist[random.randint(0,5)])
        item_url_list = list(set(item_url_list))
        return item_url_list

    def item_url2rev_url(self, item_url_list):
        rev_url_list = []
        option = '/ref=cm_cr_arp_d_paging_btm_2?ie=UTF8&pageNumber={}'
        for url in item_url_list:
            if '/dp' in url:
                tmp = url.split('/dp')
                item_id = tmp[1].split('/')[1]
                rev_url = tmp[0] + '/product-reviews/' + item_id + option
                rev_url_list.append(rev_url)
        return rev_url_list

    def get_rev(self, rev_url):
        tlist = [1,3,5,7,9,11]
        value_xpath = 'string(//*[@id="cm_cr-review_list"]/div[{rev_no}]/div[@class="a-section celwidget"]/div[1]/a[1]/i/span)'
        title_xpath = 'string(//*[@id="cm_cr-review_list"]/div[{rev_no}]/div[@class="a-section celwidget"]/div[1]/a[2])'
        date_xpath = 'string(//*[@id="cm_cr-review_list"]/div[{rev_no}]/div[@class="a-section celwidget"]/div[2]/span[@class="a-size-base a-color-secondary review-date"])'
        cmt_xpath = 'string(//*[@id="cm_cr-review_list"]/div[{rev_no}]/div[@class="a-section celwidget"]/div[4]/span)'
        vote_xpath = 'string(//*[@id="cm_cr-review_list"]/div[{rev_no}]/div[@class="a-section celwidget"]/div[5]/div/span[1]/div[1]/span)'
        def _value(rev_html, rev_no):
            return int(self.CC.get_xpath(rev_html, value_xpath.format(rev_no=rev_no))[-3])
        def _title(rev_html, rev_no):
            return mojimoji.zen_to_han(str(self.CC.get_xpath(rev_html, title_xpath.format(rev_no=rev_no))), kana=False).lower()
        def _date(rev_html, rev_no):
            return str(self.CC.get_xpath(rev_html, date_xpath.format(rev_no=rev_no))).replace('年','-').replace('月','-').replace('日','')
        def _comment(rev_html, rev_no):
            return mojimoji.zen_to_han(str(self.CC.get_xpath(rev_html, cmt_xpath.format(rev_no=rev_no))), kana=False).lower()
        def _vote(rev_html, rev_no):
            tmp = str(self.CC.get_xpath(rev_html, vote_xpath.format(rev_no=rev_no)))
            if '人' in tmp:
                return int(tmp.split('人')[0].replace(',',''))
            else:
                return 0

        item_id = rev_url.split('/product-reviews/')[1].split('/')[0]
        i = 0
        flg = True
        review_list = []
        while flg:
            i += 1
            rev_html = self.CC.get_html(rev_url.format(i))
            time.sleep(tlist[random.randint(0,5)])
            for rev_no in range(1,11):
                comment = _comment(rev_html, rev_no)
                if comment == '':
                    flg = False
                    break
                else:
                    value = _value(rev_html, rev_no)
                    title = _title(rev_html, rev_no)
                    date = _date(rev_html, rev_no)
                    vote = _vote(rev_html, rev_no)
                    review = [item_id, value, title, date, vote, comment]
                    review_list.append(review)
            if flg:
                print("{}:{}    OK".format(item_id, i))
            else:
                print("{}:{}    NG".format(item_id, i))

        return review_list

class Crawler_RakutenReview(CrawlerCore):

    def __init__(self):
        super().__init__()
        self.CC = CrawlerCore()
        df = pd.read_csv("review_allnum.txt", "\t", header=None)
        self.rev_allnum = {}
        for i in range(len(df)):
            self.rev_allnum[df.iloc[i,0]] = int(df.iloc[i,1])

    #レビュー本文の取得
    def get_revMain(self, rev_html, rev_no):
        rev_cmt_xpath = 'string(//*[@id="revRvwSec"]/div[1]/div/div[@class="revRvwUserSecCnt"]/div[{}]/div[@class="revRvwUserMain"]/div[@class="revUserEntry"]/div[1]/dl/dd[@class="revRvwUserEntryCmt description"])'.format(rev_no)
        rev_cmt = self.CC.get_xpath(rev_html, rev_cmt_xpath)
        return str(rev_cmt)

    #レビュー点数の取得
    def get_revValue(self, rev_html, rev_no):
        rev_value_xpath = 'string(//*[@id="revRvwSec"]/div[1]/div/div[@class="revRvwUserSecCnt"]/div[{}]/div[@class="revRvwUserMain"]/p[@class="revRvwUserMainHead rating"]/span[@class="revUserRvwerNum value"])'.format(rev_no)
        rev_value = self.CC.get_xpath(rev_html, rev_value_xpath)
        if rev_value == '':
            return 0
        else:
            return int(rev_value)

    #レビューの投稿日の取得
    def get_revDate(self, rev_html, rev_no):
        rev_date_xpath = 'string(//*[@id="revRvwSec"]/div[1]/div/div[@class="revRvwUserSecCnt"]/div[{}]/div[@class="revRvwUserMain"]/p[@class="revRvwUserMainHead rating"]/span[@class="revUserEntryDate dtreviewed"])'.format(rev_no)
        rev_date = self.CC.get_xpath(rev_html, rev_date_xpath)
        return str(rev_date)

    #参考になったと回答した人数の取得
    def get_revAnsNum(self, rev_html, rev_no):
        rev_ansnum_xpath = 'string(//*[@id="revRvwSec"]/div[1]/div/div[@class="revRvwUserSecCnt"]/div[{}]/div[@class="revRvwUserMain"]/p[@class="revUserEntryAns"]/span[@class="revEntryAnsTxt"]/span[@class="revEntryAnsNum"])'.format(rev_no)
        rev_ansnum = self.CC.get_xpath(rev_html, rev_ansnum_xpath)
        if rev_ansnum == '':
            rev_ansnum = 0
        else:
            rev_ansnum = int(rev_ansnum.replace(",",""))
        return rev_ansnum

    def get_comment(self, rev_html):
        comment = self.CC.get_xpath(rev_html, '//*[@class="revRvwUserSec hreview"]//*[@class="revRvwUserEntryCmt description"]')
        comment = [str(cmt.text_content()) for cmt in comment]
        return comment
    
    def get_value(self, rev_html):
        value = self.CC.get_xpath(rev_html, '//*[@class="revRvwUserSec hreview"]//*[@class="revUserRvwerNum value"]')
        value = [float(v.text_content()) for v in value]
        return value

    def get_date(self, rev_html):
        date = self.CC.get_xpath(rev_html, '//*[@class="revRvwUserSec hreview"]//*[@class="revUserEntryDate dtreviewed"]')
        date = [str(d.text_content()) for d in date]
        return date

    def get_vote(self, rev_html):
        vote = self.CC.get_xpath(rev_html, '//*[@class="revRvwUserSec hreview"]//*[@class="revEntryAnsNum"]')
        vote = [int(v.text_content().replace(",","")) for v in vote]
        return vote

    #ページ内全てのレビューを取得
    def get_rev1page(self, rev_html):
        comment = self.get_comment(rev_html)
        value = self.get_value(rev_html)
        date = self.get_date(rev_html)
        vote = self.get_vote(rev_html)
        vote.extend([0 for i in range(len(comment)-len(vote))])
        rev_list = []
        for i in range(len(comment)):
            try:
                rev_list.append([value[i], date[i], vote[i], comment[i]])
            except:
                rev_list = []
        return rev_list

    #メイン
    def crawler_main(self, item_url):
        if not os.path.exists("review_html/"+item_url):
            #商品ページのHTMLを取得
            url = "https://item.rakuten.co.jp/" + item_url
            item_html = self.CC.get_html(url)

            #レビューページのURL取得
            rev_url_xpath = '//*[@id="js-review-widget"]/tr/td/table/tr/td/table/tr/td[2]/a[1]/@href'
            rev_url = self.CC.get_xpath(item_html, rev_url_xpath)
            if rev_url != []:
                rev_url = rev_url[0]
            else:
                return []

            rev_html = self.CC.get_html(rev_url)
            base_url = rev_url.split('1.1')[0] + '{}.1/sort3/?l2-id=review_PC_il_search_03'

            #レビュー総数およびページ数の取得
            rev_allnum_xpath = 'string(//*[@id="revRvwSec"]/div[1]/div/div[@class="revRvwUserSecCnt"]/div[@class="revPagerSec"]/p)'
            rev_allnum = self.CC.get_xpath(rev_html, rev_allnum_xpath)
            rev_allnum = int(rev_allnum.split(' ')[1][:-2].replace(",", ""))
            rev_pagenum = rev_allnum // 15 + 1
            rev_pagenum = min(rev_pagenum, 100)

            with open("review_allnum.txt", "a") as f:
                f.write("{}\t{}\n".format(item_url, rev_allnum))

            os.makedirs("review_html/"+item_url)
        
        else:
            rev_allnum = self.rev_allnum[item_url]
            rev_pagenum = rev_allnum // 15 + 1
            rev_pagenum = min(rev_pagenum, 100)

        rev_list = []

        #レビューデータ取得
        for rev_page_no in range(1, rev_pagenum+1):
            #rev_allnum -= 15
            #rev_num = min(15, rev_allnum)
            fname = "review_html/"+item_url+"{}.pkl".format(rev_page_no)
            if not os.path.exists(fname):
                rev_html = self.CC.get_html(base_url.format(rev_page_no))
                #htmlファイル保存
                with open(fname, "wb") as f:
                    pickle.dump(rev_html, f, protocol=4)
            else:
                rev_html = pickle.load(open(fname, "rb"))
            rev_list += self.get_rev1page(rev_html)

        return rev_list


def rakuten_crawler(item_url_list):
    Rakuten = Crawler_RakutenReview()

    #データベースの設定
    table_name = 'review'
    db_settings = {
        "host":"localhost",
        "database":"database",
        "user":"user",
        "password":"password",
        "port":3306,
        "charset":"utf8mb4"
    }

    con = mysql.connector.connect(**db_settings)
    cur = con.cursor()

    #テーブルの作成
    """
    cur.execute('DROP TABLE IF EXISTS {}'.format(table_name))
    sql = 'CREATE TABLE {} (item_url TEXT, value FLOAT, rev_date DATE, vote INT, comment TEXT, get_date DATE)'.format(table_name)
    cur.execute(sql)
    """

    if not os.path.exists("review_html"):
        os.mkdir("review_html")

    for item_url in item_url_list:

        rev_list = Rakuten.crawler_main(item_url=item_url)
        today = datetime.date.today().strftime("%Y-%m-%d")
        if len(rev_list) > 0:
            sql = 'INSERT INTO {} VALUES(%s, %s, %s, %s, %s, %s)'.format(table_name)
            rev_list = [[item_url]+rev+[today] for rev in rev_list]
            num_data = len(rev_list)
            buf = int(math.ceil(num_data/128))
            for i in range(buf):
                data = rev_list[i*128:(i+1)*128]
                cur.executemany(sql, data)
                con.commit()
            print("{}   Completed".format(item_url))
        else:
            print("{}   no review".format(item_url))
    cur.close()
    con.close()

def amazon_get_item_list():
    urls = ['https://www.amazon.co.jp/s/ref=lp_2151982051_pg_2?rh=n%3A2127209051%2Cn%3A%212127210051%2Cn%3A2151982051&page={}&ie=UTF8&qid=1533796822',
            'https://www.amazon.co.jp/s/ref=lp_2151970051_pg_2?rh=n%3A2127209051%2Cn%3A%212127210051%2Cn%3A2151970051&page={}&ie=UTF8&qid=1533796970',
            'https://www.amazon.co.jp/s/ref=lp_2151984051_pg_2?rh=n%3A2127209051%2Cn%3A%212127210051%2Cn%3A2151984051&page={}&ie=UTF8&qid=1533797178']
    n_page = [400, 400, 400]
    amazon = Crawler_AmazonReview()
    item_url_list = []
    for i, url in enumerate(urls):
        item_url_list += amazon.get_item_url_list(url, n_page[i])
    item_url_list = list(set(item_url_list))
    return item_url_list

def amazon_crawler():
    amazon = Crawler_AmazonReview()
    with open("amazon_pc_sub.pkl","rb") as f:
        rev_url_list = pickle.load(f)

    #データベースの設定
    table_name = 'review'
    db_settings = {
        "host":"localhost",
        "database":"database",
        "user":"user",
        "password":"password",
        "port":3306,
        "charset":"utf8mb4"
    }

    con = mysql.connector.connect(**db_settings)
    cur = con.cursor()

    #テーブルの作成
    #cur.execute('DROP TABLE IF EXISTS {}'.format(table_name))
    #sql = 'CREATE TABLE {} (item_id TEXT, value INT, title TEXT, date DATE, vote INT, comment TEXT)'.format(table_name)
    #cur.execute(sql)

    for rev_url in rev_url_list:
        item_id = rev_url.split('/product-reviews/')[1].split('/')[0]

        rev_list = amazon.get_rev(rev_url)
        if len(rev_list) > 0:
            sql = 'INSERT INTO {} VALUES(%s, %s, %s, %s, %s, %s)'.format(table_name)
            cur.executemany(sql, rev_list)
            con.commit()
            print("{}   Completed".format(item_id))
        else:
            print("{}   no review".format(item_id))

    cur.close()
    con.close()


if __name__=='__main__':
    #商品URLの読込
    with open("list.pkl", "rb") as f:
        item_url_list = pickle.load(f)
    L = item_url_list[int(sys.argv[1]):int(sys.argv[2])]
    rakuten_crawler(L)
