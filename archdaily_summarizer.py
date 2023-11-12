import argparse
import sys
import time
import re
import miniaudio
import audiofile
import numpy as np

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from keys import OPENAI_KEY, HUGGING_FACE_KEY

from cleantext import clean
import validators


client = OpenAI(
    api_key=OPENAI_KEY,
    # personal
    organization='org-IHBVjd2Q6O0VkeAlTmot35YV',
)



def partition_string(message):
    max_length = 4096
    lines = message.split('. ')
    partitions = []
    current_partition = ''

    for line in lines:
        if len(current_partition + line) <= max_length:
            current_partition += line + '. '
        else:
            partitions.append(current_partition)
            current_partition = line + '. '

    if current_partition:
        partitions.append(current_partition)

    return partitions


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default=None, type=str)
    parser.add_argument('--audify', default=False, type=bool)
    parser.add_argument('--chatid', default='', type=str)
    return parser.parse_args()

def clean_text(text):
    text= clean(text=text,
                fix_unicode=True,
                to_ascii=True,
                lower=False,
                no_line_breaks=True,
                no_urls=True,
                no_emails=True,
                no_phone_numbers=False,
                no_numbers=False,
                no_digits=False,
                no_currency_symbols=True,
                no_punct=False,
                lang='en',
                )
    pattern = r'[#$%&\"()*+,/:;<=>@[\]^_`{|}~©️]'
    text = re.sub(pattern, '', text)
    text = text.replace('Save this picture!', '')
    text = text.replace('Text description provided by the architects.', '')
    

    return text

def grab_text_content(soup):
    text_content = []
    for data in soup.find_all('p'):
        text = data.get_text().strip()
        text = clean_text(text)
        split = text.split(' ')
        if len(split) > 5 : # remove short text
            if text not in text_content:
                text_content.append(text)
    text_content = ' '.join(text_content)
    return text_content

def get_summary(input_text, wordcount=150):
    guide = f'You are a helpful assistant to summarize the following text into below {wordcount} words summary.'
    try:
        response = client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    temperature=0,
                    # giving history of chat
                    messages=[
                        {'role': 'system', 'content': guide},
                        {'role': 'assistant', 'content': 'Give me the text and I will summarize it'},
                        {'role': 'user', 'content': input_text },
                    ]
                )
        return(response.choices[0].message.content)
    except:
        return ''
    
def hierarchical_summary(text):
    partitions = partition_string(text)
    while (len(partitions) > 1):
        sum = ''
        for partition in partitions:
            sum += get_summary(partition)
        partitions = partition_string(sum)
    
    return get_summary('\n'.join(partitions), wordcount=100)
    

def response(input):
    if validators.url(input):
        try:
            response = requests.get(input)
            if response.status_code == 200:
                html_content = response.content
            else:
                return f'Error retrieving HTML content:{response.status_code}' 
            soup = BeautifulSoup(html_content, 'html.parser')
            text = grab_text_content(soup)
            summary = hierarchical_summary(text)
            return summary
        except:
            return 'failed parsing the url'
    else:
        return 'not a valid url'
    
    
def audify(text, chat_id):
    API_URL = 'https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech'
    headers = {'Authorization': f'Bearer {HUGGING_FACE_KEY}'}
    response = requests.post(API_URL, headers=headers, json={'inputs':text})
    audio = response.content
    # print(response.content)
    audio_path = f'temp/summary_{chat_id}.wav'
    decoded = miniaudio.decode(audio, sample_rate = 16000)
    miniaudio.wav_write_file(audio_path, decoded)
    # with open(audio_path, 'wb') as f:
    #     f.write(audio)
    return audio_path


if __name__ == '__main__':
    opts = get_args()
    output = response(opts.url)
    if not opts.audify:
        b = bytes(output, 'utf-8')
        sys.stdout.buffer.write(b)
    else:
        chat_id = opts.chatid
        output = audify(output, chat_id)
        b = bytes(output, 'utf-8')
        sys.stdout.buffer.write(b)