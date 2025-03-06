import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import argparse

def create_table(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            title TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    return (conn, cursor)
def crawl_openstack_page(url, cursor, conn, visited):
    base_link=url[:url.rfind('/')+1]
    if url in visited:
        #print('already visited!')
        return True
    visited.add(url)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return False

        soup = BeautifulSoup(response.text, 'lxml')
  
        title = soup.title.string.strip() if soup.title else "No Title"
        if not 'OpenStack End User Guide' in title:
            content = ' '.join([p.get_text() for p in soup.find_all(['p', 'div'], class_="docs-body")])

            cursor.execute('INSERT INTO documents (url, title, content) VALUES (?, ?, ?)', (url, title, content))
            conn.commit()

            print(f"Crawled: {title}, url: {url}")
   
        for link in soup.find_all('a',attrs={"class":"reference internal"}):
            href = link['href']
            if '#' in href:
                continue
            new_base_link=base_link
            while '..' in href:
                last_lslash = new_base_link.rfind('/')
                new_base_link = new_base_link[:new_base_link.rfind('/',0,last_lslash)+1]
                href=href[3:]
            new_url = new_base_link+href
            if not crawl_page(new_url, cursor, conn, visited):
                print(url, new_url)

        time.sleep(1)  # 서버 부하 방지를 위한 딜레이

    except Exception as e:
        print(f"Error crawling {url}: {e}")
    return True
def crawl_kube_page(url, cursor, conn, visited):
    '''
    Code from GPT
    import requests
    from bs4 import BeautifulSoup
    import json
    import time

    # Kubernetes 문서 홈페이지 URL
    BASE_URL = "https://kubernetes.io"
    DOCS_URL = "https://kubernetes.io/docs/home/"

    # 요청 헤더 (크롤링 차단 방지)
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    def get_documentation_links():
        """ Kubernetes 문서 페이지에서 모든 문서 링크를 가져오는 함수 """
        response = requests.get(DOCS_URL, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        
        doc_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/docs/") and not href.endswith("/docs/home/"):  # 홈 페이지 제외
                full_url = BASE_URL + href
                doc_links.append(full_url)
        
        return list(set(doc_links))  # 중복 제거 후 반환

    def scrape_documentation_page(url):
        """ 개별 문서 페이지를 크롤링하여 제목과 본문을 추출하는 함수 """
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No Title"
        content = "\n".join([p.get_text(strip=True) for p in soup.find_all("p")])  # 모든 <p> 태그 텍스트 추출

        return {"url": url, "title": title, "content": content}

    def main():
        """ 전체 문서를 크롤링하고 JSON 파일로 저장하는 메인 함수 """
        documentation_links = get_documentation_links()
        print(f"🔍 Found {len(documentation_links)} documentation pages.")

        scraped_data = []
        for idx, doc_url in enumerate(documentation_links, 1):
            print(f"📄 [{idx}/{len(documentation_links)}] Scraping: {doc_url}")
            try:
                doc_data = scrape_documentation_page(doc_url)
                scraped_data.append(doc_data)
                time.sleep(1)  # 서버 부하 방지
            except Exception as e:
                print(f"⚠️ Error scraping {doc_url}: {e}")

        # JSON 파일로 저장
        with open("kubernetes_docs.json", "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, ensure_ascii=False, indent=4)

        print("✅ Scraping complete! Data saved in 'kubernetes_docs.json'.")

    '''
    
    base_link=url[:url.rfind('/')+1]
    if url in visited:
        print('already visited!')
        return True
    visited.add(url)

    try:
        response = requests.get(url, timeout=1
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return False

        soup = BeautifulSoup(response.text, 'lxml')
  
        title = soup.title.string.strip() if soup.title else "No TitlefS"
        content = ' '.join([p.get_tex() for p in soup.find_all(['p', 'div'], class_="docs-body")])

        cursor.execute('INSERT INTO documents (url, title, content) VALUES (?, ?, ?)', (url, title, content))
        conn.commit()

        print(f"Crawled: {title}, url: {url}")
   
        for link in soup.find_all('a',attrs={"class":"reference internal"}):
            href = link['href']
            if '#' in href:
                continue
            new_base_link=base_link
            while '..' in href:
                last_lslash = new_base_link.rfind('/')
                new_base_link = new_base_link[:new_base_link.rfind('/',0,last_lslash)+1]
                href=href[3:]
            new_url = new_base_link+href
            if not crawl_page(new_url, cursor, conn, visited):
                print(url, new_url)

        time.sleep(1)  # 서버 부하 방지를 위한 딜레이

    except Exception as e:
        print(f"Error crawling {url}: {e}")
    return True
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openstack', action='store_true', help='Crawl OpenStack docs')
    parser.add_argument('--openstacksdk', action='store_true', help='Crawl OpenStack SDK docs')
    parser.add_argument('--kubernetes', action='store_true', help='Crawl Kubernetes docs')
    parser.add_argument('--ansible', action='store_true', help='Crawl Ansible docs')
    parser.add_argument('--all', action='store_true', help='Crawl all docs')
    args = parser.parse_args()
    if args.openstack or args.all:
        # OpenStack docs crawling
        conn, cursor = create_table('openstack_docs.db')
        start_url = "https://docs.openstack.org/mitaka/user-guide/index.html"
        visited_urls = set()
        crawl_openstack_page(start_url, cursor, conn, visited_urls)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.openstacksdk or args.all:
        conn, cursor = create_table('openstacksdk_docs.db')
        start_url = "https://docs.openstack.org/openstacksdk/latest/user/index.html"
        visited_urls = set()
        crawl_page(start_url, cursor, conn, visited_urls)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.kubernetes or args.all:
        conn, cursor = create_table('kubernetes_docs.db')
        start_url = "https://kubernetes.io/docs/home/"
        visited_urls = set()
        crawl_page(start_url, cursor, conn, visited_urls)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.ansible or args.all:    
        conn, cursor = create_table('ansible_docs.db')
        start_url = "https://docs.ansible.com/"
        visited_urls = set()
        crawl_page(start_url, cursor, conn, visited_urls)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()