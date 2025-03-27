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
            if not crawl_openstack_page(new_url, cursor, conn, visited):
                print(url, new_url)

        time.sleep(1)

    except Exception as e:
        print(f"Error crawling {url}: {e}")
    return True
def crawl_kube_page(url, cursor, conn, visited):    
    if url in visited:
        print('already visited!')
        return True
    visited.add(url)

    try:
        response = requests.get(url, timeout=1)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return False

        soup = BeautifulSoup(response.text, 'lxml')
  
        title = soup.title.string.strip() if soup.title else "No Title"
        content = ' '.join([p.get_text() for p in soup.find_all(['p', 'div'], class_="td-content")])
        if not content=='':
            #print('\n#####################################')
            #print(title)
            #print(content)

            cursor.execute('INSERT INTO documents (url, title, content) VALUES (?, ?, ?)', (url, title, content))
            conn.commit()

            print(f"Crawled: {title}, url: {url}")
   
        for ul in soup.find_all('ul'):
            for item in ul.find_all('li'):
                a_tag = item.find('a')
                if a_tag:
                    href = a_tag.get('href')
                    if not 'https://' in href or not 'docs' in href:
                        continue
                    
                    if not crawl_kube_page(href, cursor, conn, visited):
                        print(url, href)

        time.sleep(1) 

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
        crawl_openstack_page(start_url, cursor, conn, visited_urls)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.kubernetes or args.all:
        conn, cursor = create_table('kubernetes_docs.db')
        start_url = "https://kubernetes.io/docs/home/"
        visited_urls = set()
        crawl_kube_page(start_url, cursor, conn, visited_urls)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.ansible or args.all:    
        conn, cursor = create_table('ansible_docs.db')
        start_url = "https://docs.ansible.com/"
        visited_urls = set()
        crawl_page(start_url, cursor, conn, visited_urls)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()