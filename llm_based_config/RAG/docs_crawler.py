import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import argparse
from stackapi import StackAPI

def create_table(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            title TEXT,
            question TEXT,
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

def crawl_ansible_page(url, cursor, conn, visited_urls):
    base_link = 'https://docs.ansible.com/ansible/latest/'
    response = requests.get(url, timeout=1)

    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return False

    soup = BeautifulSoup(response.text, 'lxml')
    links = soup.find_all('a',attrs={"class":"reference internal"})
    for link in links:
        link = link.get('href')
        if 'roadmap' in link:
            continue
        if link in visited_urls:
            continue
        visited_urls.add(link)
        #print('Visiting '+ base_link+link)
        success=False
        for _ in range(3):
            try:
                response = requests.get(base_link+link, timeout=1)
                success=True
                break
            except:
                time.sleep(1)
                continue
        if not success or response.status_code != 200:
            print(f"Failed to fetch {base_link+link}")
            continue
        soup = BeautifulSoup(response.text, 'lxml')

        big_title = soup.title.string.strip() if soup.title else "No Title"
        article_div = soup.find("div", attrs={"itemprop": "articleBody"})
        if article_div:
            sections = article_div.find_all("section", id=True)
            for section in sections:
                title = section['id']
                filtered_elements = section.find_all(['p', 'div'])
                content=  "\n".join(str(elem.get_text()) for elem in filtered_elements)
                #print('###################')
                #print(big_title+' / '+ title)
                #print(content)
                cursor.execute('INSERT INTO documents (url, title, content) VALUES (?, ?, ?)', (url, big_title+' / '+title, content))
                conn.commit()

        print(f"Crawled: {title}, url: {base_link+link}")
        time.sleep(1.5)     
    return True

def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def crawl_stackoverflow(cursor, conn):
    SITE = StackAPI('stackoverflow')
    SITE.page_size = 100
    SITE.max_pages = 20 # need fix? banned from stack oveflow
    
    keyword = 'Kubernetes'
    # Todo: Maybe use multiple keywords later

    questions_data = SITE.fetch('search/advanced', q=keyword, filter='withbody')
    questions = questions_data.get('items', [])
    
    docs_num=0 
    for question in questions:
        if question.get('view_count', 0) < 500:
            continue

        q_id = question['question_id']
        q_title = question.get('title', '')
        q_body = html_to_text(question.get('body', ''))

        answers_data = SITE.fetch('questions/{ids}/answers', ids=[q_id], filter='withbody')
        answers = answers_data.get('items', [])

        # Filtering the answers. Only remain answers with 2 or more votes
        filtered_answers = []
        for answer in answers:
            score = answer.get('score', 0)
            accepted = answer.get('is_accepted', False)
            if score >= 2 or accepted:
                filtered_answers.append(answer)

        if not filtered_answers:
            #There is no appropriate answer.
            continue

        total_answer=''
        for answer in filtered_answers:
            total_answer+=html_to_text(answer.get('body', ''))
        cursor.execute('INSERT INTO documents (title, question, content) VALUES (?, ?, ?)', (q_title, q_body, total_answer))
        conn.commit()
        print(f"Crawled: {q_title}")
        docs_num+=1
        time.sleep(1)
    
    return docs_num

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openstack', action='store_true', help='Crawl OpenStack docs')
    parser.add_argument('--openstacksdk', action='store_true', help='Crawl OpenStack SDK docs')
    parser.add_argument('--kubernetes', action='store_true', help='Crawl Kubernetes docs')
    parser.add_argument('--ansible', action='store_true', help='Crawl Ansible docs')
    parser.add_argument('--stack', action='store_true', help='Crawl StackOverflow')
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
        visited_urls = set()
        start_url = "https://docs.ansible.com/ansible/latest/index.html"
        crawl_ansible_page(start_url, cursor, conn,visited_urls)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.stack or args.all:    
        conn, cursor = create_table('stackoverflow_docs.db')   
        visited_urls = set()
        results = crawl_stackoverflow(cursor, conn)
        print(f'Total {results} nums of doc. crawled.')
        conn.close()