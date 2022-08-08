import threading

from selenium.webdriver.common.by import By
from selenium import webdriver
from urllib.request import urlopen
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os
import socket

socket.setdefaulttimeout(15)

search_word = "푸들"

# driver = webdriver.Chrome("C:/chromedriver/chromedriver.exe")  # 크롬드라이브 다운로드 경로
# driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")  # Get명령을 통해 해당 url 접속
# # 위에 링크는 이미지 검색 페이지
# elem = driver.find_element_by_name("q")  # 검색창을 탐색
# elem.send_keys(search_word)  # 검색할 단어 입력
# elem.send_keys(Keys.RETURN)  # 검색 시작

SCROLL_PAUSE_TIME = 1.5

search = ["Irish terrier", "아이리시 테리어"]

store = ['087.Irish_terrier', '087.Irish_terrier']


# driver.get("https://www.naver.com/")
# elem = driver.find_element_by_name("q")  # 검색창을 탐색
# elem.send_keys(search_word)  # 검색할 단어 입력
# elem.send_keys(Keys.RETURN)  # 검색 시작

def timeout(limit_time):  # timeout
    start = time.time()
    while True:
        if time.time() - start > limit_time or SAVE_FLAG:
            raise Exception('timeout. or image saved.')


for i in range(0, 22):
    if i % 2 == 0:
        cnt = 1
    driver = webdriver.Chrome("C:/Users/as/PycharmProjects/pythonProject4/chromedriver.exe")  # 크롬드라이브 다운로드 경로
    driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")  # Get명령을 통해 해당 url 접속
    # 위에 링크는 이미지 검색 페이지
    elem = driver.find_element("name", "q")  # 검색창을 탐색
    elem.send_keys(search[i])  # 검색할 단어 입력
    elem.send_keys(Keys.RETURN)  # 검색 시작

    SAVE_FLAG = False

    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element("CSS_SELECTOR", ".mye4qd").click() #find_element_by_css_selector(".mye4qd").click()
            except:
                break
        last_height = new_height
    # 해당 부분은 스크롤을 내려서 더 많은 검색을 진행할지 버튼 찾아서 눌러줌

    images = driver.find_element(By.CSS_SELECTOR, '.rg_i.Q4LuWd')  # 이미지 썸네일들 찾음

    for image in images:
        SAVE_FLAG = False
        timer = threading.Thread(target=timeout, args=(30,))
        try:
            image.click()
            time.sleep(1)

            # 'img.n3VNCb'가 여러 개 있기 때문에 더 세부적으로 지정 필요

            xpath_ = '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img'
            bigImage_url = driver.find_element(By.XPATH, "xpath_").get_attribute('src')
            file_ext = bigImage_url.split('.')[-1]

            # 이미지 확장자가 있는 것과 없는 것을 구분하여 저장
            if file_ext in ['jpg', 'jpeg', 'webp', 'png']:
                filename = os.path.basename(bigImage_url)  # 파일명만 추출

                # 파일을 디렉토리에 저장
                urllib.request.urlretrieve(bigImage_url, "./data/" + store[i] + "/" + str(cnt) + "." + file_ext)

                cnt += 1
                if cnt % 5 == 0:
                    print(f'검색어 "{search[i]}"의 이미지 {cnt}장 저장 중...')
            else:
                # print(bigImage_url)  # “data:image/jpeg;base64,/9j/4AAQSkZJR....”
                filename = str(time.time()) + '_.jpg'
                urllib.request.urlretrieve(bigImage_url, "./data/" + store[i] + "/" + str(cnt) + ".jpg")
                SAVE_FLAG = True
                cnt += 1
                if timer.is_alive():
                    timer.join()

        except Exception as err:
            print(err)
    driver.close()
# for image in images:
#     try:
#         image.click()  # 이미지 클릭
#         time.sleep(2)  # 로딩을 생각해서 2초의 대기
#         imgUrl = driver.find_element_by_xpath(
#             "/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img").get_attribute(
#             'src')
#         urllib.request.urlretrieve(imgUrl, "bulldog" + str(count) + ".jpg")
#         count = count + 1
#     except:
#         pass
