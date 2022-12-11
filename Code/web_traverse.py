import pandas as pd
import os
import bs4
from tqdm import tqdm
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from socket import gaierror
from utils import avail_data

code_dir = os.getcwd()
data_dir = os.path.join(os.path.split(code_dir)[0], 'Data')
avail_data(data_dir)
final_data = pd.read_json(os.path.join(data_dir, r'Combined_Headlines.json'))
sarcastic_data = final_data.iloc[final_data.filter(final_data.is_sarcastic == 1).index]


def return_text(url):
    offset = 30
    try:
        source = urlopen(url).read()
        soup = bs4.BeautifulSoup(source, 'lxml')
    except (URLError, gaierror):
        try:
            new_url = url[offset:]
            source = urlopen(new_url).read()
            soup = bs4.BeautifulSoup(source, 'lxml')
        except (URLError, gaierror, ValueError, HTTPError):
            text = ""
            return text
    allowlist = ["p", "em", "i", "b"]
    blocklist = ["Sign Up", "HuffPost", "Huffington"]
    text_elements = list()
    for t in soup.find_all(text=True):
        if t.parent.name in allowlist:
            contains_blocked = False
            for block in blocklist:
                if t.find(block) != -1:
                    contains_blocked = True
            if not contains_blocked:
                text_elements.append(t)

    text = " ".join(text_elements)
    return text


sarcastic_data["body"] = [""] * len(sarcastic_data)
print("\nScraping TheOnion articles...")
with tqdm(total=len(sarcastic_data)) as pbar:
    for i, row in sarcastic_data.iterrows():
        body = return_text(row[0])
        sarcastic_data.loc[i, "body"] = body
        pbar.update()

output_path = os.path.join(data_dir, "sarcastic_output.json")
sarcastic_data.to_json(output_path)
