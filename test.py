import difflib



if __name__ == "__main__":
    url1 = "http://service.library.mtime.com"
    url2 = "http://service.library.mtime.co"

    d = difflib.Differ()
    diff = d.compare(url1.splitlines(), url2.splitlines())
    print(list(diff))
    # print('\n'.join(list(diff)))

