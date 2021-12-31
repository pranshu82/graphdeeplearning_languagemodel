from bs4 import BeautifulSoup
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from embedding_utils import read_glove_vecs

nltk.download('stopwords')

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#convert to lower
def convert_lower_case(text):
    return text.lower()

#function for removing special characters
def remove_special_characters(text):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

#Stemming the text
def simple_stemmer(text):
    ps=PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

stop=set(stopwords.words('english'))
tokenizer=ToktokTokenizer()
stopword_list=nltk.corpus.stopwords.words('english')

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#preprocess text
def preprocess_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_special_characters(text)
    #text = simple_stemmer(text)
    text = remove_stopwords(text)
    text = convert_lower_case(text)
    return text
