from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.common.by import By
import re
from bs4 import BeautifulSoup
import os
from selenium.webdriver.chrome.options import Options
import json
from tqdm import tqdm


def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options)
    driver.get(
        "https://www.law.go.kr/LSW/lsAstSc.do?menuId=391&subMenuId=397&tabMenuId=437&query=#AJAX"
    )
    time.sleep(3)
    return driver


def get_row_per_page():
    page = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "contentBody"))
    )

    header = page.find_element(By.ID, "conTop").find_element(By.TAG_NAME, "h2").text
    page_source = page.get_attribute("innerHTML")
    조문정보_html = re.search(
        r"<!--\s*조문정보\s*-->[\s\S]*?<!--\s*조문정보\s*-->", page_source, re.DOTALL
    )
    soup = BeautifulSoup(조문정보_html.group(0), "html.parser")
    pgroup = soup.find_all("div", {"class": "pgroup"})
    lawcon = [
        x.get_text(strip=True) for x in pgroup if x.find("div", {"class": "lawcon"})
    ]
    current_url = driver.current_url

    data = {"title": header, "contents": lawcon, "url": current_url}
    os.makedirs(f"./datas", exist_ok=True)
    with open(f"./datas/{header}.json", mode="w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    try:
        driver = get_driver()

        # 형사법 페이지 이동 / 예정법령은 제외
        driver.find_element(By.ID, "liLsFd").find_element(By.XPATH, "parent::a").click()
        time.sleep(2)
        driver.find_element(By.ID, "divLsFd").find_element(By.ID, "lsFd09").click()
        time.sleep(2)
        driver.find_element(By.ID, "efCheck").click()
        time.sleep(2)
        driver.execute_script("setEfChk(); return;")

        paging = driver.find_element(By.CLASS_NAME, "paging").find_elements(
            By.TAG_NAME, "li"
        )
        # 페이지네이션
        for p_i in tqdm(range(1, len(paging)), desc="페이지네이션"):
            main_window = driver.current_window_handle
            count = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody tr"))
            )

            for i in tqdm(range(0, len(count)), desc="데이터 추출"):
                current_row = WebDriverWait(driver, 10).until(
                    EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "tbody tr"))
                )
                row = current_row[i].find_element(By.CSS_SELECTOR, "td.ctn1 a")
                row.click()

                # 팝업으로 drive 이동
                for window in driver.window_handles:
                    if window != main_window:
                        driver.switch_to.window(window)
                        break

                # 데이터 추출
                get_row_per_page()

                # 팝업 닫기 및 메인 창 복귀
                driver.close()
                driver.switch_to.window(main_window)
                time.sleep(1)

            paging_area = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "paging"))
            )
            pages = paging_area.find_elements(By.TAG_NAME, "li")
            pages[p_i].click()
            time.sleep(5)
    finally:
        driver.close()
