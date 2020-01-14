import os
import json
import requests
from bs4 import BeautifulSoup
from bs4 import Comment
import re
import urllib


def scrape_caption_text(url):
    post_id = url[url.find(".com/")+5:]
    page = requests.get(url)
    bsPage = BeautifulSoup(page.content, 'html.parser')
    comments=bsPage.find_all(string=lambda text:isinstance(text,Comment))
    if len(comments) == 0:
        return {'post_text': None, 'image_description': None, 'link_header': None}    
    bsPost = BeautifulSoup(comments[0], 'html.parser')
    #text from post
    post_text = ""
    for p in bsPost.findAll("p"):
        post_text += p.getText() + "\n" 
    
    #download image file
    image_description=""
    imageDiv = bsPost.find_all("div", class_="_3x-2")
    if len(imageDiv) >0:
        imgs = imageDiv[0].find_all("img")
        if len(imgs) >0:
            image_url = imgs[0]["src"]
            urllib.urlretrieve(image_url,"./Data/ImagesNew/"+post_id+".jpg")
            try:
                image_description = imgs[0]["alt"][18:]
            except:
                pass
        
    #link header
    linkRX = re.compile(r'target="_blank"[^<]+>([^<]+)[</a>]')
    link_result = linkRX.findall(comments[0])
    link_header = None if len(link_result) == 0 else link_result[0]
    
    return {'post_text': post_text, 'image_description': image_description, 'link_header': link_header}

if __name__ == '__main__':

    post_ids_to_scrape = #load file with post ids into an array here

    scraped_text = {}
    for post_id in post_ids_to_scrape:
        scraped_text[post_id] = scrape_caption_text("www.facebook.com/"+post_id) 

    with open('./Data/scraped_posts.json', 'w') as fp:
        json.dump(scraped_text, fp)
