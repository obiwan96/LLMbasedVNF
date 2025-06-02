import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import sqlite3
import time
import argparse
import json
import os
from stackapi import StackAPI
import sys


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
def crawl_openstack_page(url, cursor, conn, visited, logging=False):
    base_link=url[:url.rfind('/')+1]
    if url in visited:
        if logging:
            print('already visited!')
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

            if logging:
                print('##################')
                print(title)
                print(content)

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
            if not crawl_openstack_page(new_url, cursor, conn, visited, logging):
                print(url, new_url)

        time.sleep(1)

    except Exception as e:
        print(f"Error crawling {url}: {e}")
    return True

def crawl_kube_page(url, cursor, conn, visited, logging=False):
    if url in visited:
        if logging:
            print('already visited!')
        return True
    visited.add(url)

    try:
        response = requests.get(url, timeout=1)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return False

        soup = BeautifulSoup(response.text, 'lxml')

        # Let's put full a document and sub-documents together.
  
        big_title = soup.title.string.strip() if soup.title else "No Title"
        content = ' '.join([p.get_text() for p in soup.find_all(['p', 'div'], class_="td-content")])
        if not content =='':
            if logging:
                print('\n#####################################')
                print(big_title)
                print(content[:30])
            cursor.execute('INSERT INTO documents (url, title, content) VALUES (?, ?, ?)', (url, big_title, content))
            conn.commit()

            # Let's find small documents
            content = soup.find_all(class_="td-content")
            if content != []:
                current_title = None
                current_paragraph = []
                for elem in content[0].descendants:
                    if elem.name in ["h2", "h3"]:
                        if current_title and current_paragraph:
                            title=big_title+'/'+current_title
                            content="\n".join(current_paragraph)
                            if logging:
                                print('\n#####################################')
                                print(title)
                                print(content[:50])
                            cursor.execute('INSERT INTO documents (url, title, content) VALUES (?, ?, ?)', (url, title, content))
                            conn.commit()
                            print(f"Crawled: {title}")
                            current_paragraph = []
                        current_title = elem.get_text(strip=True)

                    elif elem.name in ["p", "code", "li", "dd", "dfn", "pre", "strong", "em"]:
                        text = elem.get_text(strip=True, separator=" ")
                        if text:
                            current_paragraph.append(text)

                    elif isinstance(elem, NavigableString):
                        text = elem.strip()
                        if text:
                            current_paragraph.append(text)

                if current_title and current_paragraph:
                    title=big_title+'/'+current_title
                    content="\n".join(current_paragraph)
                    if logging:
                        print('\n#####################################')
                        print(title)
                        print(content[:50])
                    cursor.execute('INSERT INTO documents (url, title, content) VALUES (?, ?, ?)', (url, title, content))
                    conn.commit()
                    print(f"Crawled: {title}")
   
        for ul in soup.find_all('ul'):
            for item in ul.find_all('li'):
                a_tag = item.find('a')
                if a_tag:
                    href = a_tag.get('href')
                    if not 'https://' in href or not 'docs' in href:
                        continue
                    
                    if not crawl_kube_page(href, cursor, conn, visited, logging):
                        print(url, href)

        time.sleep(1.5) 

    except Exception as e:
        print(f"Error crawling {url}: {e}")
    return True

def crawl_ansible_page(url, cursor, conn, visited_urls, logging=False):
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
        if logging:
            print('Visiting '+ base_link+link)
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
                if logging:
                    print('###################')
                    print(big_title+' / '+ title)
                    print(content)
                cursor.execute('INSERT INTO documents (url, title, content) VALUES (?, ?, ?)', (url, big_title+' / '+title, content))
                conn.commit()

        print(f"Crawled: {title}, url: {base_link+link}")
        time.sleep(1.5)     
    return True

def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def crawl_stackoverflow(cursor, conn, take_up = False, logging=False):
    SITE = StackAPI('stackoverflow')
    SITE.page_size = 100
    SITE.max_pages = 20 # need modify? banned from stack oveflow
    
    # State file for saving last question date
    state_file = "stack_crawl_state.json"
    if os.path.exists(state_file) and take_up:
        # Load the last crawl state
        with open(state_file, "r") as f:
            state = json.load(f)
            last_timestamp = state.get("last_creation_date", 0)
            max_last_timestamp = last_timestamp
            crawled_ids = set(state.get("crawled_ids", []))
    else:
        last_timestamp = 0
        max_last_timestamp = 0
        crawled_ids = set()
    #keyword = 'Kubernetes'
    print('Please input the keyword to search:')
    keyword = input()
    print('ok.')
    # Todo: Maybe use multiple keywords later

    page = 1
    while True:
        try:
            questions_data = SITE.fetch('search/advanced', q=keyword, filter='withbody', fromdate=last_timestamp, page=page)
            if not questions_data['items']:
                break
            questions = questions_data.get('items', [])
            if not questions:
                print("No more questions.")
                break
            for question in questions:
                q_id = question['question_id']
                if q_id in crawled_ids or question.get('view_count', 0) < 500:
                    continue
                crawled_ids.add(q_id)
                max_last_timestamp = max(max_last_timestamp, question['creation_date'])
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
                if logging:
                    print('####################')
                    print(q_title)
                    print('--------------------')
                    print(q_body)
                    print('Answer:')
                    print(total_answer)
                cursor.execute('INSERT INTO documents (title, question, content) VALUES (?, ?, ?)', (q_title, q_body, total_answer))
                conn.commit()
                print(f"Crawled: {q_title}")
            if not questions_data.get("has_more", False):
                break

            page += 1
            time.sleep(1)

        except Exception as e:
            print(f"Error: {e}")
            break
    with open(state_file, "w") as f:
        json.dump({
            "last_creation_date": max_last_timestamp,
            "crawled_ids": list(crawled_ids)
        }, f)
    
    return len(crawled_ids)

def delete_overlap(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if 'stackoverflow' in db_name:
        cursor.execute('''
            DELETE FROM documents
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM documents
                GROUP BY title
            )
        ''')
    else:
        cursor.execute('''
            DELETE FROM documents
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM documents
                GROUP BY url, title
            )
        ''')
    deleted_rows = cursor.rowcount
    print(f"Deleted {deleted_rows} duplicate rows from {db_name}.")
    conn.commit()
    conn.close()

def input_several_lines(prompt):
    print("\033[92m"+prompt+' Input break with ctrl + D. \033[0m')
    text = sys.stdin.read()
    return text

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openstack', action='store_true', help='Crawl OpenStack docs')
    parser.add_argument('--openstacksdk', action='store_true', help='Crawl OpenStack SDK docs')
    parser.add_argument('--kubernetes', action='store_true', help='Crawl Kubernetes docs')
    parser.add_argument('--ansible', action='store_true', help='Crawl Ansible docs')
    parser.add_argument('--stack', action='store_true', help='Crawl StackOverflow')
    parser.add_argument('--all', action='store_true', help='Crawl all docs')
    parser.add_argument('--take-up', action='store_true', help='Take up the last crawl state')
    parser.add_argument('--logging', action='store_true', help='Logging or not')
    parser.add_argument('--manual-add', action='store_true', help='Manually add documents to the database')
    
    args = parser.parse_args()
    if args.manual_add:
        db_name = input("\033[91mEnter the database name ('open', 'k8s', 'ansible', 'stack') or type 'exit' to quit: \033[0m")
        if db_name.lower() == 'open':
            db_name = 'openstack_docs.db'
        elif db_name.lower() == 'k8s':
            db_name = 'kubernetes_docs.db'
        elif db_name.lower() == 'ansible':
            db_name = 'ansible_docs.db'
        elif db_name.lower() == 'stack':
            db_name = 'stackoverflow_docs.db'
        else:
            print("\033[91mInvalid database name. Exiting.\033[0m")
            sys.exit(1)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        while True:
            title = input("\033[92mEnter the title of the document: \033[0m")
            if title.lower() == 'exit':
                conn.close()
                break
            if db_name == 'stackoverflow_docs.db':
                question = input_several_lines("Enter the question of the document: ")
            else:
                url = input("\033[92mEnter the URL of the document: \033[0m")
            content = input_several_lines("Enter the content of the document: ")
            if db_name == 'stackoverflow_docs.db':
                cursor.execute('INSERT INTO documents (title, question, content) VALUES (?, ?, ?)', (title, question, content))
            else:
                cursor.execute('INSERT INTO documents (url, title, content) VALUES (?, ?, ?)', (url, title, content))
            confirm = input("\033[92mDo you really want to commit? (y/n): \033[0m")
            if confirm.lower() != 'y':
                print("\033[91mOperation cancelled.\033[0m")
                continue
            conn.commit()
            print(f"\033[91mDocument added to {db_name}.\033[0m")
        conn.close()
    if args.openstack or args.all:
        # OpenStack docs crawling
        conn, cursor = create_table('openstack_docs.db')
        start_url = "https://docs.openstack.org/mitaka/user-guide/index.html"
        visited_urls = set()
        crawl_openstack_page(start_url, cursor, conn, visited_urls, args.logging)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.openstacksdk or args.all:
        conn, cursor = create_table('openstacksdk_docs.db')
        start_url = "https://docs.openstack.org/openstacksdk/latest/user/index.html"
        visited_urls = set()
        crawl_openstack_page(start_url, cursor, conn, visited_urls, args.logging)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.kubernetes or args.all:
        conn, cursor = create_table('kubernetes_docs.db')
        start_url = "https://kubernetes.io/docs/home/"
        visited_urls = set()
        crawl_kube_page(start_url, cursor, conn, visited_urls, args.logging)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.ansible or args.all:    
        conn, cursor = create_table('ansible_docs.db')   
        visited_urls = set()
        start_url = "https://docs.ansible.com/ansible/latest/index.html"
        crawl_ansible_page(start_url, cursor, conn,visited_urls, args.logging)
        print(f'Total {len(visited_urls)} nums of doc. crawled.')
        conn.close()
    if args.stack or args.all:    
        # for stack overflow, make in several times
        # conn, cursor = create_table('stackoverflow_docs.db')   
        conn = sqlite3.connect('stackoverflow_docs.db')
        cursor = conn.cursor()
        visited_urls = set()
        results = crawl_stackoverflow(cursor, conn, args.take_up, args.logging)
        print(f'Total {results} nums of doc. crawled.')
        conn.close()