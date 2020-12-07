import urllib.request

def download_file(download_url, filename):
    response = urllib.request.urlopen(download_url)    
    file = open(filename + ".pdf", 'wb')
    file.write(response.read())
    file.close()

for i in range(13, 24): 
    pdf_path = f"http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-{i}.pdf"
    download_file(pdf_path, f"lec-{i}")
