import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

class GetDataWikiart:
    # Página base de WikiArt de Van Gogh
    BASE_URL = 'https://www.wikiart.org/'
    ARTIST_URL = 'https://www.wikiart.org/en/artist-name/all-works/text-list'
    HEADERS = {
        'User-Agent': 'Mozilla/5.0'
    }

    def __init__(self,save_dir,artist):
        self.url = self.ARTIST_URL.replace('artist-name',artist.lower().replace(' ','-'))
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir=save_dir
        self.artist = artist

    def download_pages(self,max_num_images,num_page_from,num_page_to):
        print(f"🔍 Obteniendo enlaces de pinturas de {self.artist}...")
        links = self.get_painting_links(max_num_images,num_page_from,num_page_to)
        print(f"🎨 {len(links)} enlaces encontrados. Iniciando descarga...")
        self.download_images(links)
        print("✅ Descarga completada.")

    def get_painting_links(self,max_num_images,num_page_from,num_page_to):
        links = []
        for page in range(num_page_from, num_page_to):
            url = f'{self.url}/{page}' if page > 1 else self.url
            response = requests.get(url, headers=self.HEADERS)
            if response.status_code != 200:
                break
            soup = BeautifulSoup(response.text, 'html.parser')
            painting_tags = soup.select('li.painting-list-text-row a')
            if not painting_tags:
                break
            for tag in painting_tags:
                link = self.BASE_URL + tag['href']
                links.append(link)
                if len(links) >= max_num_images:
                    break
            if len(links) >= max_num_images:
                break
        return links

    def download_images(self,links):
        filename = self.artist.lower().replace(' ','-')
        for i, link in enumerate(tqdm(links, desc="Descargando pinturas")):
            try:
                response = requests.get(link, headers=self.HEADERS)
                soup = BeautifulSoup(response.text, 'html.parser')
                img_tag = soup.find('img', {'itemprop': 'image'})
                if not img_tag:
                    continue
                img_url = img_tag['src']
                img_data = requests.get(img_url, headers=self.HEADERS).content
                with open(os.path.join(self.save_dir, f'{filename}_{i:04d}.jpg'), 'wb') as f:
                    f.write(img_data)
            except Exception as e:
                print(f"Error al descargar {link}: {e}")
