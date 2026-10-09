import requests

OL_BASE = "https://openlibrary.org/search.json"

def search_books(query, max_results=5):
    params = {
        "q":      query,
        "limit":  max_results,
        "fields": "title,author_name,first_publish_year,subject,key,isbn"
    }
    try:
        resp = requests.get(OL_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[Books Agent] Open Library error: {e}")
        return []

    books = []
    for doc in data.get("docs", [])[:max_results]:
        try:
            key     = doc.get("key", "")
            isbn    = doc.get("isbn", [None])[0]
            ol_url  = f"https://openlibrary.org{key}"
            gb_url  = f"https://www.google.com/search?q={requests.utils.quote(doc.get('title',''))}+book" if not isbn else f"https://books.google.com/books?isbn={isbn}"
            subjects = doc.get("subject", [])[:4]

            books.append({
                "title":   doc.get("title", "Untitled"),
                "authors": doc.get("author_name", ["Unknown"])[:2],
                "year":    doc.get("first_publish_year"),
                "subjects": subjects,
                "ol_url":  ol_url,
                "gb_url":  gb_url,
                "isbn":    isbn
            })
        except Exception:
            continue
    return books


def get_books(query, max_results=5):
    print("[Books Agent] Searching Open Library...")
    books = search_books(query, max_results=max_results)
    print(f"[Books Agent] {len(books)} books found.")
    return books